#pragma once

#include "component/gpu_vertex_set.cuh"
#include "component/gpu_schedule.cuh"

__device__ void gpu_pattern_matching_final_in_exclusion(const GPUSchedule *schedule, GPUVertexSet *vertex_set, GPUVertexSet &subtraction_set,
                                                        GPUVertexSet &tmp_set, unsigned long long &local_ans, uint32_t *edge, uint32_t *vertex) {
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

constexpr int MAX_DEPTH = 5; // 非递归pattern matching支持的最大深度

template <int depth>
__device__ void gpu_pattern_matching_func(const GPUSchedule *schedule, GPUVertexSet *vertex_set, GPUVertexSet &subtraction_set, GPUVertexSet &tmp_set,
                                          unsigned long long &local_ans, uint32_t *edge, uint32_t *vertex) {

    if (depth == schedule->get_size() - schedule->get_in_exclusion_optimize_num()) {
        gpu_pattern_matching_final_in_exclusion(schedule, vertex_set, subtraction_set, tmp_set, local_ans, edge, vertex);
        return;
    }

    int loop_set_prefix_id = schedule->get_loop_set_prefix_id(depth);
    int loop_size = vertex_set[loop_set_prefix_id].get_size();
    if (loop_size <= 0) //这个判断可能可以删了
        return;

    uint32_t *loop_data_ptr = vertex_set[loop_set_prefix_id].get_data_ptr();
    uint32_t min_vertex = 0xffffffff;
    for (int i = schedule->get_restrict_last(depth); i != -1; i = schedule->get_restrict_next(i))
        if (min_vertex > subtraction_set.get_data(schedule->get_restrict_index(i)))
            min_vertex = subtraction_set.get_data(schedule->get_restrict_index(i));
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
        gpu_pattern_matching_func<depth + 1>(schedule, vertex_set, subtraction_set, tmp_set, local_ans, edge, vertex);
        if (depth + 1 != MAX_DEPTH) {
            if (threadIdx.x % THREADS_PER_WARP == 0)
                subtraction_set.pop_back();
            __threadfence_block();
        }
    }
}

template <>
__device__ void gpu_pattern_matching_func<MAX_DEPTH>(const GPUSchedule *schedule, GPUVertexSet *vertex_set, GPUVertexSet &subtraction_set,
                                                     GPUVertexSet &tmp_set, unsigned long long &local_ans, uint32_t *edge, uint32_t *vertex) {
    // assert(false);
}

// device global variables
__device__ unsigned long long dev_sum;
__device__ unsigned int dev_cur_task;

__global__ void gpu_pattern_matching(unsigned task_end, size_t buffer_size, uint32_t *edge_from, uint32_t *edge, uint32_t *vertex, uint32_t *tmp,
                                     const GPUSchedule *schedule) {
    __shared__ unsigned int block_edge_idx[WARPS_PER_BLOCK];
    extern __shared__ GPUVertexSet block_vertex_set[];

    int num_prefixes = schedule->get_total_prefix_num();
    int num_vertex_sets_per_warp = num_prefixes + 2;

    int wid = threadIdx.x / THREADS_PER_WARP;            // warp id within the block
    int lid = threadIdx.x % THREADS_PER_WARP;            // lane id
    int global_wid = blockIdx.x * WARPS_PER_BLOCK + wid; // global warp id
    unsigned int &edge_idx = block_edge_idx[wid];
    GPUVertexSet *vertex_set = block_vertex_set + wid * num_vertex_sets_per_warp;

    if (lid == 0) {
        edge_idx = 0;
        ptrdiff_t offset = buffer_size * global_wid * num_vertex_sets_per_warp;
        for (int i = 0; i < num_vertex_sets_per_warp; ++i) {
            vertex_set[i].set_data_ptr(tmp + offset); // 注意这是个指针+整数运算，自带*4
            offset += buffer_size;
        }
    }
    GPUVertexSet &subtraction_set = vertex_set[num_prefixes];
    GPUVertexSet &tmp_set = vertex_set[num_prefixes + 1];

    __threadfence_block();

    uint32_t v0, v1;
    uint32_t l, r;

    unsigned long long sum = 0;

    while (true) {
        if (lid == 0) {
            edge_idx = atomicAdd(&dev_cur_task, 1);
            unsigned int i = edge_idx;
            if (i < task_end) {
                subtraction_set.init();
                subtraction_set.push_back(edge_from[i]);
                subtraction_set.push_back(edge[i]);
            }
        }
        __threadfence_block();

        unsigned int i = edge_idx;
        if (i >= task_end)
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
        gpu_pattern_matching_func<2>(schedule, vertex_set, subtraction_set, tmp_set, local_sum, edge, vertex);
        sum += local_sum;
    }

    if (lid == 0) {
        atomicAdd(&dev_sum, sum);
    }
}