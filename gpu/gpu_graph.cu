#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <string>

#include "common.h"
#include "dataloader.h"
#include "graph.h"
#include "motif_generator.h"
#include "schedule_IEP.h"
#include "vertex_set.h"
#include "timeinterval.h"

#include "component/gpu_device_detect.cuh"
#include "src/gpu_pattern_matching.cuh"


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

    gpu_pattern_matching<<<num_blocks, THREADS_PER_BLOCK, context->block_shmem_size>>>(g->e_cnt, buffer_size, context);

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

    if (argc < 4) {
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

    tmpTime.check();
    
    bool is_pattern_valid;
    Schedule_IEP schedule_iep(p, is_pattern_valid, 1, 1, true, g->v_cnt, g->e_cnt, g->tri_cnt);
    if (!is_pattern_valid) {
        fprintf(stderr, "pattern is invalid!\n");
        return 1;
    }

    tmpTime.print("Schedule time cost");

    pattern_matching(g, schedule_iep);

    allTime.print("Total time cost");
    return 0;
}
