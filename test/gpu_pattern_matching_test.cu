#include "../gpu/src/gpu_pattern_matching.cuh"
#include "../include/common.h"
#include "../include/pattern.h"
#include "../include/schedule.h"
#include <../include/dataloader.h>
#include <../include/graph.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <iostream>
#include <string>

TEST(gpu_pattern_matching_test, gpu_pattern_matching_test_patents) {
    Graph *g;
    DataLoader D;
    bool ok = D.fast_load(g, "/home/cqq/data/patents.g");
    ASSERT_EQ(ok, true);

    int size[] = {5, 6, 6, 6, 7, 7};

    const char *pattern[50] = {"0111010011100011100001100",
                               "011011101110110101011000110000101000",
                               "011111101000110111101010101101101010",
                               "011110101101110000110000100001010010",
                               "0111111101111111011101110100111100011100001100000",
                               "0111111101111111011001110100111100011000001100000"};

    long long ans[] = {7375094981, 19186236035, 9600941704, 86946614984, 138737462736, 37814965911};

    for (int i = 0; i < 6; i++) {
        Pattern p(size[i], pattern[i]);
        bool is_pattern_valid = false;
        Schedule_IEP schedule_iep(p, is_pattern_valid, 1, 1, true, g->v_cnt, g->e_cnt, g->tri_cnt);
        ASSERT_EQ(is_pattern_valid, true);

        PatternMatchingDeviceContext *context;
        gpuErrchk(cudaMallocManaged((void **)&context, sizeof(PatternMatchingDeviceContext)));
        context->init(g, schedule_iep);

        uint32_t buffer_size = VertexSet::max_intersection_size;
        int max_active_blocks_per_sm;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks_per_sm, gpu_pattern_matching, THREADS_PER_BLOCK, context->block_shmem_size);
        printf("max number of active warps per SM: %d\n", max_active_blocks_per_sm * WARPS_PER_BLOCK);

        printf("iep: %lld\n", schedule_iep.get_in_exclusion_optimize_redundancy());

        unsigned long long sum = 0;

        gpu_pattern_matching<<<num_blocks, THREADS_PER_BLOCK, context->block_shmem_size>>>(g->e_cnt, buffer_size, context);

        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaMemcpy(&sum, context->dev_sum, sizeof(sum), cudaMemcpyDeviceToHost));

        sum /= schedule_iep.get_in_exclusion_optimize_redundancy();

        printf("count %llu\n", sum);
        ASSERT_EQ(sum, ans[i]);

        context->destroy();
        gpuErrchk(cudaFree(context));
    }
}