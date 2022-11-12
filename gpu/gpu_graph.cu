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

#include "component/gpu_device_detect.cuh"

#include "src/gpu_pattern_matching.cuh"

#include <timeinterval.h>

TimeInterval allTime;
TimeInterval tmpTime;



void pattern_matching(Graph *g, const Schedule_IEP &schedule_iep) {
    PatternMatchingDeviceContext *context;
    gpuErrchk(cudaMallocManaged((void **)&context, sizeof(PatternMatchingDeviceContext)));
    context->init(g, schedule_iep);

    uint32_t buffer_size = VertexSet::max_intersection_size;
    int max_active_blocks_per_sm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks_per_sm, gpu_pattern_matching, THREADS_PER_BLOCK, context->block_shmem_size);
    printf("max number of active warps per SM: %d\n", max_active_blocks_per_sm * WARPS_PER_BLOCK);

    tmpTime.print("Prepare time cost");
    tmpTime.check();

    printf("iep: %lld\n", schedule_iep.get_in_exclusion_optimize_redundancy());
    
    unsigned long long sum = 0;

    gpu_pattern_matching<<<num_blocks, THREADS_PER_BLOCK, context->block_shmem_size>>>(g->e_cnt, buffer_size, context);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(&sum, context->dev_sum, sizeof(sum), cudaMemcpyDeviceToHost));

    sum /= schedule_iep.get_in_exclusion_optimize_redundancy();


    printf("count %llu\n", sum);
    tmpTime.print("Counting time cost");

    context->destroy();
    gpuErrchk(cudaFree(context));
}

int main(int argc, char *argv[]) {
    print_device_information();
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

    pattern_matching(g, schedule_iep);

    allTime.print("Total time cost");
    return 0;
}
