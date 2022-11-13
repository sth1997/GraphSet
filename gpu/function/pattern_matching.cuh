#pragma once

#include "../component/gpu_device_context.cuh"
#include "../component/gpu_schedule.cuh"
#include "../component/gpu_vertex_set.cuh"

constexpr int MAX_DEPTH = 5; // 非递归pattern matching支持的最大深度

struct PatternMatchingDeviceContext : public GraphDeviceContext {
    GPUSchedule *dev_schedule;
    unsigned long long *dev_sum;
    unsigned long long *dev_cur_edge;
    size_t block_shmem_size;
    void init(const Graph *_g, const Schedule_IEP &schedule) {
        g = _g;
        // prefix + subtraction + tmp + extra (n-2)
        int num_vertex_sets_per_warp = schedule.get_total_prefix_num() + schedule.get_size();

        size_t size_edge = g->e_cnt * sizeof(uint32_t);
        size_t size_vertex = (g->v_cnt + 1) * sizeof(e_index_t);
        size_t size_tmp = VertexSet::max_intersection_size * num_total_warps * (schedule.get_total_prefix_num() + 2) *
                          sizeof(uint32_t); // prefix + subtraction + tmp
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
        int64_t cur_edge = 0;
        gpuErrchk(cudaMalloc((void **)&dev_cur_edge, sizeof(cur_edge)));
        gpuErrchk(cudaMemcpy(dev_cur_edge, &cur_edge, sizeof(cur_edge), cudaMemcpyHostToDevice));

        gpuErrchk(cudaMallocManaged((void **)&dev_schedule, sizeof(GPUSchedule)));
        dev_schedule->create_from_schedule(schedule);

        log("Memory Usage:\n");
        log("  Global memory usage (GB): %.3lf \n", (size_edge + size_edge + size_vertex + size_tmp) / (1024.0 * 1024 * 1024));
        log("  Shared memory for vertex set per block: %ld bytes\n",
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

/**
 * @brief 用于 vertex_induced 的计算（好像没怎么测过）
 *
 */
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

/**
 * @brief 以模板形式伪递归的计算函数
 *
 */
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

/**
 * @brief 模板递归的边界
 *
 */
template <>
__device__ void GPU_pattern_matching_func<MAX_DEPTH>(const GPUSchedule *schedule, GPUVertexSet *vertex_set, GPUVertexSet &subtraction_set,
                                                     GPUVertexSet &tmp_set, unsigned long long &local_ans, uint32_t *edge, e_index_t *vertex) {
    // assert(false);
}