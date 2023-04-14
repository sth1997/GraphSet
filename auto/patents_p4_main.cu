#include <graph.h>
#include <dataloader.h>
#include <vertex_set.h>
#include <common.h>
#include <schedule_IEP.h>
#include <motif_generator.h>

#include <cassert>
#include <cstring>
#include <cstdint>
#include <string>
#include <algorithm>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <sys/time.h>
#include <chrono>

#include "component/utils.cuh"
#include "component/gpu_device_context.cuh"
#include "component/gpu_device_detect.cuh"
#include "src/gpu_pattern_matching.cuh"
#include "timeinterval.h"


__global__ void gpu_pattern_matching_generated(e_index_t edge_num, uint32_t buffer_size, PatternMatchingDeviceContext *context);


TimeInterval allTime;
TimeInterval tmpTime;

void pattern_matching(Graph *g, const Schedule_IEP &schedule_iep) {
    tmpTime.check();
    PatternMatchingDeviceContext *context;
    gpuErrchk(cudaMallocManaged((void **)&context, sizeof(PatternMatchingDeviceContext)));
    context->init(g, schedule_iep);

    uint32_t buffer_size = VertexSet::max_intersection_size;
    int max_active_blocks_per_sm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks_per_sm, gpu_pattern_matching, THREADS_PER_BLOCK, context->block_shmem_size);
    fprintf(stderr, "max number of active warps per SM: %d\n", max_active_blocks_per_sm * WARPS_PER_BLOCK);

    tmpTime.print("Prepare time cost");
    tmpTime.check();

    unsigned long long sum = 0;

    gpu_pattern_matching_generated<<<num_blocks, THREADS_PER_BLOCK, context->block_shmem_size>>>(g->e_cnt, buffer_size, context);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(&sum, context->dev_sum, sizeof(sum), cudaMemcpyDeviceToHost));

    sum /= schedule_iep.get_in_exclusion_optimize_redundancy();

    printf("Pattern count: %llu\n", sum);
    tmpTime.print("Counting time cost");

    context->destroy();
    gpuErrchk(cudaFree(context));
}

int main(int argc, char *argv[]) {
    get_device_information();
    Graph *g;
    DataLoader D;

    if (argc < 2) {
        fprintf(stderr, "Usage: %s graph_file pattern_size pattern_string\n", argv[0]);
        return 1;
    }

    using std::chrono::system_clock;
    auto t1 = system_clock::now();

    bool ok = D.fast_load(g, argv[1]);
    if (!ok) {
        fprintf(stderr, "data load failure :-(\n");
        return 0;
    }

    auto t2 = system_clock::now();
    auto load_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    fprintf(stderr, "Load data success! time: %g seconds\n", load_time.count() / 1.0e6);

    allTime.check();

    int pattern_size = atoi(argv[2]);
    const char *pattern_str = argv[3];

    Pattern p(pattern_size, pattern_str);

    printf("pattern = ");
    p.print();

    fprintf(stderr, "Max intersection size %d\n", VertexSet::max_intersection_size);

    bool is_pattern_valid;
    Schedule_IEP schedule_iep(p, is_pattern_valid, 1, 1, true, g->v_cnt, g->e_cnt, g->tri_cnt);
    if (!is_pattern_valid) {
        fprintf(stderr, "pattern is invalid!\n");
        return 1;
    }

    pattern_matching(g, schedule_iep);

    allTime.print("Total time cost");
    return 0;
}
__global__ void gpu_pattern_matching_generated(e_index_t edge_num, uint32_t buffer_size, PatternMatchingDeviceContext *context) {
    __shared__ e_index_t block_edge_idx[WARPS_PER_BLOCK];
    extern __shared__ GPUVertexSet block_vertex_set[];
    GPUSchedule *schedule = context->dev_schedule;
    uint32_t *tmp = context->dev_tmp;
    uint32_t *edge = (uint32_t *)context->dev_edge;
    e_index_t *vertex = context->dev_vertex;
    uint32_t *edge_from = (uint32_t *)context->dev_edge_from;
    int wid = threadIdx.x / THREADS_PER_WARP, lid = threadIdx.x % THREADS_PER_WARP, global_wid = blockIdx.x * WARPS_PER_BLOCK + wid;
    e_index_t &edge_idx = block_edge_idx[wid];
    GPUVertexSet *vertex_set = block_vertex_set + wid * 7;
    if (lid == 0) {
        edge_idx = 0;
        uint32_t offset = buffer_size * global_wid * 7;
        for (int i = 0; i < 7; ++i) {
            vertex_set[i].set_data_ptr(tmp + offset);
            offset += buffer_size;
        }
    }
    GPUVertexSet& subtraction_set = vertex_set[5];
    __threadfence_block();
    uint32_t v0, v1;
    e_index_t l, r;
    unsigned long long sum = 0;
    while (true) {
        if (lid == 0) {
            edge_idx = atomicAdd(context->dev_cur_edge, 1);
            unsigned int i = edge_idx;
            if (i < edge_num) {
                subtraction_set.init();
                subtraction_set.push_back(edge_from[i]);
                subtraction_set.push_back(edge[i]);
            }
        }
        __threadfence_block();
        e_index_t i = edge_idx;
        if(i >= edge_num) break;
        v0 = edge_from[i];
        v1 = edge[i];
        get_edge_index(v0, l, r);
        if (threadIdx.x % THREADS_PER_WARP == 0)
            vertex_set[0].init(r - l, &edge[l]);
        __threadfence_block();
        if(v0 <= v1) continue;
        get_edge_index(v1, l, r);
        GPUVertexSet* tmp_vset;
        intersection2(vertex_set[1].get_data_ptr(), vertex_set[0].get_data_ptr(), &edge[l], vertex_set[0].get_size(), r - l, &vertex_set[1].size);
        if (vertex_set[1].get_size() == 0) continue;
        if (threadIdx.x % THREADS_PER_WARP == 0)
            vertex_set[2].init(r - l, &edge[l]);
        __threadfence_block();
        if (vertex_set[2].get_size() == 0) continue;
        extern __shared__ char ans_array[];
        int* ans = ((int*) (ans_array + 704)) + 3 * (threadIdx.x / THREADS_PER_WARP);
        int loop_size_depth2 = vertex_set[0].get_size();
        if( loop_size_depth2 <= 0) continue;
        uint32_t* loop_data_ptr_depth2 = vertex_set[0].get_data_ptr();
        uint32_t min_vertex_depth2 = 0xffffffff;
        for(int i_depth2 = 0; i_depth2 < loop_size_depth2; ++i_depth2) {
            uint32_t v_depth2 = loop_data_ptr_depth2[i_depth2];
            if(subtraction_set.has_data(v_depth2)) continue;
            unsigned int l_depth2, r_depth2;
            get_edge_index(v_depth2, l_depth2, r_depth2);
            {
            tmp_vset = &vertex_set[3];
            if (threadIdx.x % THREADS_PER_WARP == 0)
                tmp_vset->init(r_depth2 - l_depth2, &edge[l_depth2]);
            __threadfence_block();
            if (r_depth2 - l_depth2 > vertex_set[2].get_size())
                tmp_vset->size -= unordered_subtraction_size(*tmp_vset, vertex_set[2], -1);
            else
                tmp_vset->size = vertex_set[2].get_size() - unordered_subtraction_size(vertex_set[2], *tmp_vset, -1);
            }
            if (vertex_set[3].get_size() == 1) continue;
            {
            tmp_vset = &vertex_set[4];
            if (threadIdx.x % THREADS_PER_WARP == 0)
                tmp_vset->init(r_depth2 - l_depth2, &edge[l_depth2]);
            __threadfence_block();
            if (r_depth2 - l_depth2 > vertex_set[1].get_size())
                tmp_vset->size -= unordered_subtraction_size(*tmp_vset, vertex_set[1], -1);
            else
                tmp_vset->size = vertex_set[1].get_size() - unordered_subtraction_size(vertex_set[1], *tmp_vset, -1);
            }
            if (threadIdx.x % THREADS_PER_WARP == 0) subtraction_set.push_back(v_depth2);
            __threadfence_block();
            ans[0] = unordered_subtraction_size(vertex_set[1], subtraction_set);
            ans[1] = vertex_set[3].get_size() - 1;
            ans[2] = vertex_set[4].get_size() - 0;
            long long val;
            val = ans[0];
            val = val * ans[0];
            val = val * ans[1];
            sum += val * 1;
            val = ans[2];
            val = val * ans[0];
            sum += val * -1;
            val = ans[0];
            val = val * ans[2];
            sum += val * -1;
            val = ans[0];
            val = val * ans[1];
            sum += val * -1;
            val = ans[2];
            sum += val * 2;
            if (threadIdx.x % THREADS_PER_WARP == 0) subtraction_set.pop_back();
            __threadfence_block();
        }
    }
    if (lid == 0) atomicAdd(context->dev_sum, sum);
}
