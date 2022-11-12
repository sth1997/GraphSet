#pragma once
#include "../component/gpu_device_context.cuh"
#include "../component/gpu_schedule.cuh"
#include "../component/gpu_vertex_set.cuh"
#include "../component/utils.cuh"

#include "../function/pattern_matching.cuh"

struct PatternMatchingDeviceContext : public GraphDeviceContext {
    GPUSchedule *dev_schedule;
    unsigned long long *dev_sum;
    unsigned int *dev_cur_edge;
    size_t block_shmem_size;
    void init(const Graph *_g, const Schedule_IEP &schedule) {
        g = _g;
        // prefix + subtraction + tmp + extra (n-2)
        int num_vertex_sets_per_warp = schedule.get_total_prefix_num() + schedule.get_size();

        size_t size_edge = g->e_cnt * sizeof(uint32_t);
        size_t size_vertex = (g->v_cnt + 1) * sizeof(e_index_t);
        size_t size_tmp = VertexSet::max_intersection_size * num_total_warps *
                          (schedule.get_total_prefix_num() + 2) * sizeof(uint32_t); // prefix + subtraction + tmp
        uint32_t *edge_from = new uint32_t[g->e_cnt];
        for (uint32_t i = 0; i < g->v_cnt; ++i) {
            for (uint32_t j = g->vertex[i]; j < g->vertex[i + 1]; ++j)
                edge_from[j] = i;
        }

        gpuErrchk(cudaMalloc((void **)&dev_edge, size_edge));
        gpuErrchk(cudaMalloc((void **)&dev_edge_from, size_edge));
        gpuErrchk(cudaMalloc((void **)&dev_vertex, size_vertex));
        gpuErrchk(cudaMalloc((void **)&dev_tmp, size_tmp));

        gpuErrchk(cudaMemcpy(dev_edge, g->edge, size_edge, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(dev_edge_from, edge_from, size_edge, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(dev_vertex, g->vertex, size_vertex, cudaMemcpyHostToDevice));

        unsigned long long sum = 0;
        gpuErrchk(cudaMalloc((void **)&dev_sum, sizeof(sum)));
        gpuErrchk(cudaMemcpy(dev_sum, &sum, sizeof(sum), cudaMemcpyHostToDevice));
        unsigned int cur_edge = 0;
        gpuErrchk(cudaMalloc((void **)&dev_cur_edge, sizeof(cur_edge)));
        gpuErrchk(cudaMemcpy(dev_cur_edge, &cur_edge, sizeof(cur_edge), cudaMemcpyHostToDevice));

        gpuErrchk(cudaMallocManaged((void **)&dev_schedule, sizeof(GPUSchedule)));
        dev_schedule->create_from_schedule(schedule);

        printf("Memory Usage:\n");
        printf("  Global memory usage (GB): %.3lf \n", (size_edge + size_edge + size_vertex + size_tmp) / (1024.0 * 1024 * 1024));
        printf("  Shared memory for vertex set per block: %ld bytes\n",
               num_vertex_sets_per_warp * WARPS_PER_BLOCK * sizeof(GPUVertexSet) +
                   schedule.in_exclusion_optimize_vertex_id.size() * WARPS_PER_BLOCK * sizeof(int));

        block_shmem_size = num_vertex_sets_per_warp * WARPS_PER_BLOCK * sizeof(GPUVertexSet) +
                           schedule.in_exclusion_optimize_vertex_id.size() * WARPS_PER_BLOCK * sizeof(int);
        dev_schedule->ans_array_offset = block_shmem_size - schedule.in_exclusion_optimize_vertex_id.size() * WARPS_PER_BLOCK * sizeof(int);

        delete[] edge_from;
    }
    void destroy() {
        gpuErrchk(cudaFree(dev_edge));
        gpuErrchk(cudaFree(dev_edge_from));
        gpuErrchk(cudaFree(dev_vertex));
        gpuErrchk(cudaFree(dev_tmp));

        gpuErrchk(cudaFree(dev_schedule->adj_mat));
        gpuErrchk(cudaFree(dev_schedule->father_prefix_id));
        gpuErrchk(cudaFree(dev_schedule->last));
        gpuErrchk(cudaFree(dev_schedule->next));
        gpuErrchk(cudaFree(dev_schedule->loop_set_prefix_id));
        gpuErrchk(cudaFree(dev_schedule->restrict_last));
        gpuErrchk(cudaFree(dev_schedule->restrict_next));
        gpuErrchk(cudaFree(dev_schedule->restrict_index));

        gpuErrchk(cudaFree(dev_schedule->in_exclusion_optimize_vertex_id));
        gpuErrchk(cudaFree(dev_schedule->in_exclusion_optimize_coef));
        gpuErrchk(cudaFree(dev_schedule->in_exclusion_optimize_flag));

        gpuErrchk(cudaFree(dev_schedule));
    }
};



/**
 * @note `buffer_size`实际上是每个节点的最大邻居数量，而非所用空间大小
 */
__global__ void gpu_pattern_matching(e_index_t edge_num, uint32_t buffer_size, PatternMatchingDeviceContext *context) {
    __shared__ e_index_t block_edge_idx[WARPS_PER_BLOCK];
    //之后考虑把tmp buffer都放到shared里来（如果放得下）
    extern __shared__ GPUVertexSet block_vertex_set[];

    GPUSchedule *schedule = context->dev_schedule;
    uint32_t *tmp = context->dev_tmp;
    uint32_t *edge = (uint32_t *)context->dev_edge;
    e_index_t *vertex = context->dev_vertex;
    uint32_t *edge_from = (uint32_t *)context->dev_edge_from;

    int num_prefixes = schedule->get_total_prefix_num();
    int num_vertex_sets_per_warp = num_prefixes + 2;

    int wid = threadIdx.x / THREADS_PER_WARP;            // warp id within the block
    int lid = threadIdx.x % THREADS_PER_WARP;            // lane id
    int global_wid = blockIdx.x * WARPS_PER_BLOCK + wid; // global warp id
    e_index_t &edge_idx = block_edge_idx[wid];
    GPUVertexSet *vertex_set = block_vertex_set + wid * num_vertex_sets_per_warp;

    if (lid == 0) {
        edge_idx = 0;
        uint32_t offset = buffer_size * global_wid * num_vertex_sets_per_warp;
        for (int i = 0; i < num_vertex_sets_per_warp; ++i) {
            vertex_set[i].set_data_ptr(tmp + offset); // 注意这是个指针+整数运算，自带*4
            offset += buffer_size;
        }
    }
    GPUVertexSet &subtraction_set = vertex_set[num_prefixes];
    GPUVertexSet &tmp_set = vertex_set[num_prefixes + 1];

    __threadfence_block(); //之后考虑把所有的syncthreads都改成syncwarp

    uint32_t v0, v1;
    e_index_t l, r;

    unsigned long long sum = 0;

    while (true) {
        if (lid == 0) {
            edge_idx = atomicAdd(context->dev_cur_edge, 1);
            e_index_t i = edge_idx;
            if (i < edge_num) {
                subtraction_set.init();
                subtraction_set.push_back(edge_from[i]);
                subtraction_set.push_back(edge[i]);
            }
        }

        __threadfence_block();

        e_index_t i = edge_idx;
        if (i >= edge_num)
            break;

        // for edge in E
        v0 = edge_from[i];
        v1 = edge[i];

        bool is_zero = false;
        get_edge_index(v0, l, r);
        for (int prefix_id = schedule->get_last(0); prefix_id != -1; prefix_id = schedule->get_next(prefix_id))
            vertex_set[prefix_id].build_vertex_set(schedule, vertex_set, &edge[l], r - l, prefix_id);

        //目前只考虑pattern size>2的情况
        // start v1, depth = 1
        if (schedule->get_restrict_last(1) != -1 && v0 <= v1)
            continue;

        get_edge_index(v1, l, r);
        for (int prefix_id = schedule->get_last(1); prefix_id != -1; prefix_id = schedule->get_next(prefix_id)) {
            vertex_set[prefix_id].build_vertex_set(schedule, vertex_set, &edge[l], r - l, prefix_id);
            if (vertex_set[prefix_id].get_size() == 0 && prefix_id < schedule->get_basic_prefix_num()) {
                is_zero = true;
                break;
            }
        }
        if (is_zero)
            continue;

        unsigned long long local_sum = 0; // local sum (corresponding to an edge index)
        GPU_pattern_matching_func<2>(schedule, vertex_set, subtraction_set, tmp_set, local_sum, edge, vertex);
        sum += local_sum;
    }

    if (lid == 0) {
        atomicAdd(context->dev_sum, sum);
    }
}