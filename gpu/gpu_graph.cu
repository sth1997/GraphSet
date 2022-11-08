/**
 * 这个版本里面没有细粒度计时。有计时的在gpu_graph_with_timer.cu里面。
 * 而且计时的方式与zms版本略有区别。
 */
#include <common.h>
#include <dataloader.h>
#include <graph.h>
#include <motif_generator.h>
#include <schedule_IEP.h>
#include <vertex_set.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <string>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <chrono>
#include <sys/time.h>

#include "component/gpu_const.cuh"
#include "component/gpu_device_context.cuh"
#include "component/gpu_schedule.cuh"
#include "component/gpu_vertex_set.cuh"
#include "component/utils.cuh"
#include <timeinterval.h>

TimeInterval allTime;
TimeInterval tmpTime;

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
        size_t size_tmp = VertexSet::max_intersection_size * sizeof(uint32_t) * num_total_warps *
                          (schedule.get_total_prefix_num() + 2); // prefix + subtraction + tmp
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

        printf("shared memory for vertex set per block: %ld bytes\n",
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
 * @brief 最终层的容斥原理优化计算。
 */
__device__ void GPU_pattern_matching_final_in_exclusion(const GPUSchedule *schedule, GPUVertexSet *vertex_set, GPUVertexSet &subtraction_set,
                                                        GPUVertexSet &tmp_set, unsigned long long &local_ans, uint32_t *edge, e_index_t *vertex) {
    int last_pos = -1;
    long long val;

    extern __shared__ char ans_array[];
    int *ans = ((int *)(ans_array + schedule->ans_array_offset)) + schedule->in_exclusion_optimize_vertex_id_size * (threadIdx.x / THREADS_PER_WARP);

    for (int i = 0; i < schedule->in_exclusion_optimize_vertex_id_size; ++i) {
        if (schedule->in_exclusion_optimize_vertex_flag[i]) {
            ans[i] = vertex_set[schedule->in_exclusion_optimize_vertex_id[i]].get_size() - schedule->in_exclusion_optimize_vertex_coef[i];
        } else {
            ans[i] = unordered_subtraction_size(vertex_set[schedule->in_exclusion_optimize_vertex_id[i]], subtraction_set);
        }
    }

    for (int pos = 0; pos < schedule->in_exclusion_optimize_array_size; ++pos) {

        if (pos == last_pos + 1)
            val = ans[schedule->in_exclusion_optimize_ans_pos[pos]];
        else {
            if (val != 0)
                val = val * ans[schedule->in_exclusion_optimize_ans_pos[pos]];
        }
        if (schedule->in_exclusion_optimize_flag[pos]) {
            last_pos = pos;
            local_ans += val * schedule->in_exclusion_optimize_coef[pos];
        }
    }
}

__device__ int lower_bound(const uint32_t *loop_data_ptr, int loop_size, int min_vertex) {
    int l = 0, r = loop_size - 1;
    while (l <= r) {
        int mid = r - ((r - l) >> 1);
        if (loop_data_ptr[mid] < min_vertex)
            l = mid + 1;
        else
            r = mid - 1;
    }
    return l;
}

template <typename T>
__device__ bool binary_search(const T data[], int n, const T &target) {
    int mid, l = 0, r = n - 1;
    while (l <= r) {
        mid = (l + r) >> 1;
        if (data[mid] < target) {
            l = mid + 1;
        } else if (data[mid] > target) {
            r = mid - 1;
        } else {
            return true;
        }
    }
    return false;
}

__device__ void remove_anti_edge_vertices(GPUVertexSet &out_buf, const GPUVertexSet &in_buf, const GPUSchedule &sched,
                                          const GPUVertexSet &partial_embedding, int vp, const uint32_t *edge, const e_index_t *vertex) {

    __shared__ uint32_t block_out_offset[THREADS_PER_BLOCK];
    __shared__ uint32_t block_out_size[WARPS_PER_BLOCK];

    auto d_out = out_buf.get_data_ptr();
    auto d_in = in_buf.get_data_ptr();
    int n_in = in_buf.get_size();

    int warp = threadIdx.x / THREADS_PER_WARP;
    int lane = threadIdx.x % THREADS_PER_WARP;
    auto out_offset = block_out_offset + warp * THREADS_PER_WARP;
    auto &out_size = block_out_size[warp];

    if (lane == 0)
        out_size = 0;

    for (int nr_done = 0; nr_done < n_in; nr_done += THREADS_PER_WARP) {
        int i = nr_done + lane;
        bool produce_output = false;

        if (i < n_in) {
            produce_output = true;
            for (int u = 0; u < vp; ++u) {
                if (sched.adj_mat[u * sched.get_size() + vp] != 0)
                    continue;

                auto v = partial_embedding.get_data(u);
                e_index_t v_neighbor_begin, v_neighbor_end;
                get_edge_index(v, v_neighbor_begin, v_neighbor_end);
                int m = v_neighbor_end - v_neighbor_begin; // m = |N(v)|

                if (binary_search(&edge[v_neighbor_begin], m, d_in[i])) {
                    produce_output = false;
                    break;
                }
            }
        }
        out_offset[lane] = produce_output;
        __threadfence_block();

#pragma unroll
        for (int s = 1; s < THREADS_PER_WARP; s *= 2) {
            uint32_t v = lane >= s ? out_offset[lane - s] : 0;
            out_offset[lane] += v;
            __threadfence_block();
        }

        if (produce_output) {
            uint32_t offset = out_offset[lane] - 1;
            d_out[out_size + offset] = d_in[i];
        }

        if (lane == THREADS_PER_WARP - 1)
            out_size += out_offset[THREADS_PER_WARP - 1];
    }

    if (lane == 0)
        out_buf.init(out_size, d_out);
    __threadfence_block();
}

constexpr int MAX_DEPTH = 5; // 非递归pattern matching支持的最大深度

template <int depth>
__device__ void GPU_pattern_matching_func(const GPUSchedule *schedule, GPUVertexSet *vertex_set, GPUVertexSet &subtraction_set, GPUVertexSet &tmp_set,
                                          unsigned long long &local_ans, uint32_t *edge, e_index_t *vertex) {

    if (depth == schedule->get_size() - schedule->get_in_exclusion_optimize_num()) {
        GPU_pattern_matching_final_in_exclusion(schedule, vertex_set, subtraction_set, tmp_set, local_ans, edge, vertex);
        return;
    }

    int loop_set_prefix_id = schedule->get_loop_set_prefix_id(depth);
    auto vset = &vertex_set[loop_set_prefix_id];
    int loop_size = vset->get_size();
    if (loop_size <= 0) //这个判断可能可以删了
        return;

    auto loop_data_ptr = vset->get_data_ptr();
    uint32_t min_vertex = 0xffffffff;
    for (int i = schedule->get_restrict_last(depth); i != -1; i = schedule->get_restrict_next(i))
        if (min_vertex > subtraction_set.get_data(schedule->get_restrict_index(i)))
            min_vertex = subtraction_set.get_data(schedule->get_restrict_index(i));

    // can be optimized via code generation
    if (schedule->is_vertex_induced) {
        GPUVertexSet &diff_buf = vertex_set[schedule->get_total_prefix_num() + depth];
        remove_anti_edge_vertices(diff_buf, vertex_set[loop_set_prefix_id], *schedule, subtraction_set, depth, edge, vertex);
        loop_data_ptr = diff_buf.get_data_ptr();
        loop_size = diff_buf.get_size();
        vset = &diff_buf;
    }

    if (depth == schedule->get_size() - 1 && schedule->get_in_exclusion_optimize_num() == 0) {
        /*
        for (int i = 0; i < loop_size; ++i)
        {
            uint32_t x = vset->get_data(i);
            bool flag = true;
            for (int j = 0; j < subtraction_set.get_size(); ++j)
                if (subtraction_set.get_data(j) == x)
                    flag = false;
            if (flag && threadIdx.x == 0)
                printf("%d %d %d %d\n", subtraction_set.get_data(0), subtraction_set.get_data(1), subtraction_set.get_data(2), x);
        }
        return;*/
        int size_after_restrict = lower_bound(loop_data_ptr, loop_size, min_vertex);
        // int size_after_restrict = -1;
        local_ans += unordered_subtraction_size(*vset, subtraction_set, size_after_restrict);
        return;
    }
    for (int i = 0; i < loop_size; ++i) {
        uint32_t v = loop_data_ptr[i];
        if (min_vertex <= v)
            break;
        if (subtraction_set.has_data(v))
            continue;
        unsigned int l, r;
        get_edge_index(v, l, r);
        bool is_zero = false;
        for (int prefix_id = schedule->get_last(depth); prefix_id != -1; prefix_id = schedule->get_next(prefix_id)) {
            vertex_set[prefix_id].build_vertex_set(schedule, vertex_set, &edge[l], r - l, prefix_id);
            if (vertex_set[prefix_id].get_size() == schedule->get_break_size(prefix_id)) {
                is_zero = true;
                break;
            }
        }
        if (is_zero)
            continue;
        if (depth + 1 != MAX_DEPTH) {
            if (threadIdx.x % THREADS_PER_WARP == 0)
                subtraction_set.push_back(v);
            __threadfence_block();
        }
        GPU_pattern_matching_func<depth + 1>(schedule, vertex_set, subtraction_set, tmp_set, local_ans, edge, vertex);
        if (depth + 1 != MAX_DEPTH) {
            if (threadIdx.x % THREADS_PER_WARP == 0)
                subtraction_set.pop_back();
            __threadfence_block();
        }
    }
}

template <>
__device__ void GPU_pattern_matching_func<MAX_DEPTH>(const GPUSchedule *schedule, GPUVertexSet *vertex_set, GPUVertexSet &subtraction_set,
                                                     GPUVertexSet &tmp_set, unsigned long long &local_ans, uint32_t *edge, e_index_t *vertex) {
    // assert(false);
}

/**
 * @note `buffer_size`实际上是每个节点的最大邻居数量，而非所用空间大小
 */
__global__ void gpu_pattern_matching(e_index_t edge_num, uint32_t buffer_size, PatternMatchingDeviceContext *context) {
    __shared__ e_index_t block_edge_idx[WARPS_PER_BLOCK]; //用int表示边之后在大图上一定会出问题！
    //之后考虑把tmp buffer都放到shared里来（如果放得下）
    extern __shared__ GPUVertexSet block_vertex_set[];

    GPUSchedule *schedule = context->dev_schedule;
    uint32_t *tmp = context->dev_tmp;
    uint32_t *edge = (uint32_t *)context->dev_edge;
    e_index_t *vertex = context->dev_vertex;
    uint32_t *edge_from = (uint32_t *)context->dev_edge_from;

    int num_prefixes = schedule->get_total_prefix_num();
    int num_vertex_sets_per_warp = num_prefixes + schedule->get_size();

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
        // GPU_pattern_matching_aggressive_func(schedule, vertex_set, subtraction_set, tmp_set, local_sum, 2, edge, vertex);
        sum += local_sum;
    }

    if (lid == 0) {
        atomicAdd(context->dev_sum, sum);
    }
}

void pattern_matching_init(Graph *g, const Schedule_IEP &schedule_iep) {

    PatternMatchingDeviceContext *context;

    gpuErrchk(cudaMallocManaged((void **)&context, sizeof(PatternMatchingDeviceContext)));

    context->init(g, schedule_iep);

    uint32_t buffer_size = VertexSet::max_intersection_size;

    int max_active_blocks_per_sm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks_per_sm, gpu_pattern_matching, THREADS_PER_BLOCK, context->block_shmem_size);
    printf("max number of active warps per SM: %d\n", max_active_blocks_per_sm * WARPS_PER_BLOCK);

    tmpTime.print("Prepare time cost");
    tmpTime.check();

    unsigned long long sum = 0;

    gpu_pattern_matching<<<num_blocks, THREADS_PER_BLOCK, context->block_shmem_size>>>(g->e_cnt, buffer_size, context);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(&sum, context->dev_sum, sizeof(sum), cudaMemcpyDeviceToHost));

    sum /= schedule_iep.get_in_exclusion_optimize_redundancy();

    printf("count %llu\n", sum);
    tmpTime.print("Counting time cost");
    //之后需要加上cudaFree

    context->destroy();
    gpuErrchk(cudaFree(context));
}

int main(int argc, char *argv[]) {
    Graph *g;
    DataLoader D;

    if (argc < 2) {
        printf("Usage: %s graph_file pattern_size pattern_string\n", argv[0]);
        return 0;
    }

    using std::chrono::system_clock;
    auto t1 = system_clock::now();

    bool ok = D.fast_load(g, argv[1]);
    if (!ok) {
        printf("data load failure :-(\n");
        return 0;
    }

    auto t2 = system_clock::now();
    auto load_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    printf("Load data success! time: %g seconds\n", load_time.count() / 1.0e6);
    fflush(stdout);

    allTime.check();

    // const char *pattern_str = "0111010011100011100001100"; // 5 house p1
    // const char *pattern_str = "011011101110110101011000110000101000"; // 6 p2
    // const char *pattern_str = "0111111101111111011101110100111100011100001100000"; // 7 p5
    // const char *pattern_str = "0111111101111111011001110100111100011000001100000"; // 7 p6

    int pattern_size = atoi(argv[2]);
    const char *pattern_str = argv[3];

    Pattern p(pattern_size, pattern_str);

    printf("pattern = \n");
    p.print();
    printf("max intersection size %d\n", VertexSet::max_intersection_size);

    bool use_in_exclusion_optimize = true;
    bool is_pattern_valid;
    Schedule_IEP schedule_iep(p, is_pattern_valid, 1, 1, use_in_exclusion_optimize, g->v_cnt, g->e_cnt, g->tri_cnt);
    if (!is_pattern_valid) {
        printf("pattern is invalid!\n");
        return 0;
    }

    pattern_matching_init(g, schedule_iep);

    allTime.print("Total time cost");
    return 0;
}
