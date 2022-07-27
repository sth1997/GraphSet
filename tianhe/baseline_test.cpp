#include <../include/graph.h>
#include "../include/graph_d.h"
#include <../include/labeled_graph.h>
#include <../include/dataloader.h>
#include "../include/pattern.h"
#include "../include/schedule_IEP.h"
#include "../include/common.h"
#include "../include/motif_generator.h"
#include "../include/vertex_set.h"

#include <mpi.h>
#include <assert.h>
#include <iostream>
#include <string>
#include <algorithm>
#include <chrono>

int main(int argc,char *argv[]) {
    
    Graph *g;
    DataLoader D;
    
    using std::chrono::system_clock;
    auto t1 = system_clock::now();

    bool ok;
    ok = D.fast_load(g, argv[1]);

    if (!ok) {
        printf("data load failure :-(\n");
        return 0;
    }

    auto t2 = system_clock::now();
    auto load_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    printf("Load data success! time: %g seconds\n", load_time.count() / 1.0e6);
    fflush(stdout);

    // const char *pattern_str = "0111010011100011100001100"; // 5 house p1
    //const char *pattern_str = "011011101110110101011000110000101000"; // 6 p2
    // const char *pattern_str = "0111111101111111011101110100111100011100001100000"; // 7 p5
    // const char *pattern_str = "0111111101111111011001110100111100011000001100000"; // 7 p6

//load graph_d
    int provided;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &provided);

    Graph_D* g_d;
    g_d=new Graph_D();
    g_d->init(g);

    int pattern_size = atoi(argv[2]);
    const char* pattern_str= argv[3];

    Pattern p(pattern_size, pattern_str);
    printf("pattern = \n");
    p.print();
    printf("max intersection size %d\n", VertexSet::max_intersection_size);
    bool is_pattern_valid;
    bool use_in_exclusion_optimize = true;
    Schedule_IEP schedule_iep(p, is_pattern_valid, 1, 1, use_in_exclusion_optimize, g->v_cnt, g->e_cnt, g->tri_cnt);
    Schedule_IEP schedule(p, is_pattern_valid, 1, 1, use_in_exclusion_optimize, g->v_cnt, g->e_cnt, g->tri_cnt); // schedule is only used for getting redundancy
    schedule_iep.set_in_exclusion_optimize_redundancy(schedule.get_in_exclusion_optimize_redundancy());

    if (!is_pattern_valid) {
        printf("pattern is invalid!\n");
        return 0;
    }
    
    double count_t1 = get_wall_time();
    int thread_count = 24;
    long long ans = g->pattern_matching(schedule_iep, thread_count);
    double count_t2 = get_wall_time();
    printf("couting time= %.6lf s\n", count_t2 - count_t1);
    printf("ans=%lld\n", ans);

    MPI_Finalize();
    return 0;
}
/* 这个是原来的
#include <../include/graph.h>
#include <../include/labeled_graph.h>
#include <../include/dataloader.h>
#include "../include/pattern.h"
#include "../include/schedule_IEP.h"
#include "../include/common.h"
#include "../include/motif_generator.h"
#include "../include/vertex_set.h"

#include <assert.h>
#include <iostream>
#include <string>
#include <algorithm>
#include <chrono>

int main(int argc,char *argv[]) {
    
    Graph *g;
    DataLoader D;
    
    using std::chrono::system_clock;
    auto t1 = system_clock::now();

    bool ok;
    ok = D.fast_load(g, argv[1]);

    if (!ok) {
        printf("data load failure :-(\n");
        return 0;
    }

    auto t2 = system_clock::now();
    auto load_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    printf("Load data success! time: %g seconds\n", load_time.count() / 1.0e6);
    fflush(stdout);

    // const char *pattern_str = "0111010011100011100001100"; // 5 house p1
    //const char *pattern_str = "011011101110110101011000110000101000"; // 6 p2
    // const char *pattern_str = "0111111101111111011101110100111100011100001100000"; // 7 p5
    // const char *pattern_str = "0111111101111111011001110100111100011000001100000"; // 7 p6

    int pattern_size = atoi(argv[2]);
    const char* pattern_str= argv[3];

    Pattern p(pattern_size, pattern_str);
    printf("pattern = \n");
    p.print();
    printf("max intersection size %d\n", VertexSet::max_intersection_size);
    bool is_pattern_valid;
    bool use_in_exclusion_optimize = true;
    Schedule_IEP schedule_iep(p, is_pattern_valid, 1, 1, use_in_exclusion_optimize, g->v_cnt, g->e_cnt, g->tri_cnt);
    Schedule_IEP schedule(p, is_pattern_valid, 1, 1, use_in_exclusion_optimize, g->v_cnt, g->e_cnt, g->tri_cnt); // schedule is only used for getting redundancy
    schedule_iep.set_in_exclusion_optimize_redundancy(schedule.get_in_exclusion_optimize_redundancy());

    if (!is_pattern_valid) {
        printf("pattern is invalid!\n");
        return 0;
    }
    
    double count_t1 = get_wall_time();
    int thread_count = 24;
    long long ans = g->pattern_matching(schedule_iep, thread_count);
    double count_t2 = get_wall_time();
    printf("couting time= %.6lf s\n", count_t2 - count_t1);
    printf("ans=%lld\n", ans);

    return 0;
}
*/