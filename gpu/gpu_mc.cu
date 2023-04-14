/**
 * 这个版本里面没有细粒度计时。有计时的在gpu_graph_with_timer.cu里面。
 * 而且计时的方式与zms版本略有区别。
 */
#include <cassert>
#include <cstring>
#include <cstdint>
#include <string>
#include <algorithm>

#include "graph.h"
#include "dataloader.h"
#include "vertex_set.h"
#include "common.h"
#include "schedule_IEP.h"
#include "motif_generator.h"
#include "timeinterval.h"

#include "component/utils.cuh"
#include "component/gpu_schedule.cuh"
#include "component/gpu_vertex_set.cuh"
#include "function/pattern_matching.cuh"
#include "src/gpu_pattern_matching.cuh"

TimeInterval allTime;
TimeInterval tmpTime;

// same as gpu_graph
double pattern_matching(Graph *g, const Schedule_IEP &schedule_iep) {
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
    double counting_time = tmpTime.print("Counting time cost");

    context->destroy();
    gpuErrchk(cudaFree(context));
    return counting_time;
}


int main(int argc,char *argv[]) {
    Graph *g;
    DataLoader D;

    using std::chrono::system_clock;
    auto t1 = system_clock::now();

    bool ok = D.fast_load(g, argv[1]);

    if (argc < 3) {
        fprintf(stderr, "Usage: %s graph_file pattern_size\n", argv[0]);
        return 1;
    }

    if (!ok) {
        fprintf(stderr, "data load failure :-(\n");
        return 1;
    } 

    auto t2 = system_clock::now();
    auto load_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    fprintf(stderr, "Load data success! time: %g seconds\n", load_time.count() / 1.0e6);

    allTime.check();

    int pattern_size = atoi(argv[2]);

    printf("motif_size: %d\n", pattern_size);

    double total_counting_time = 0.0;

    MotifGenerator mg(pattern_size);
    std::vector<Pattern> motifs = mg.generate();
    printf("Motifs number = %d\n", motifs.size());
    for (int i = 0; i < motifs.size(); ++i) {
        Pattern p(motifs[i]);

        printf("pattern = ");
        p.print();
    
        fprintf(stderr, "max intersection size %d\n", VertexSet::max_intersection_size);
        
        bool is_pattern_valid;
        bool use_in_exclusion_optimize = true;
        Schedule_IEP schedule_iep(p, is_pattern_valid, 1, 1, true, g->v_cnt, g->e_cnt, g->tri_cnt);

        if (!is_pattern_valid) {
            fprintf(stderr, "pattern is invalid!\n");
            return 1;
        }

        total_counting_time += pattern_matching(g, schedule_iep);

    }
    printf("Total *COUNTING* time: %.6lf\n", total_counting_time);
    allTime.print("Total time cost");
    return 0;
}