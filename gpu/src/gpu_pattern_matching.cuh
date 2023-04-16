#pragma once
#include "../component/gpu_device_context.cuh"
#include "../component/gpu_schedule.cuh"
#include "../component/gpu_vertex_set.cuh"
#include "../component/utils.cuh"

#include "../function/pattern_matching.cuh"

/**
 * @note `buffer_size`实际上是每个节点的最大邻居数量，而非所用空间大小
 *
 * 当用作 MPI 的时候，edge_num 指的是结束的边的编号，此时 dev_cur_edge 初始值并不为 0.
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
    // extra n-2 vertex_sets for vertex_induced (motif counting)
    if(schedule->is_vertex_induced) num_vertex_sets_per_warp += schedule->get_size() - 2;

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

        //目前只考虑pattern size>2的情况
        // start v1, depth = 1
        if (schedule->get_restrict_last(1) != -1 && v0 <= v1)
            continue;

        bool is_zero = false;
        get_edge_index(v0, l, r);
        for (int prefix_id = schedule->get_last(0); prefix_id != -1; prefix_id = schedule->get_next(prefix_id))
            vertex_set[prefix_id].build_vertex_set(schedule, vertex_set, &edge[l], r - l, prefix_id);

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