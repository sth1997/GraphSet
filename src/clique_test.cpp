#include <../include/graph.h>
#include <../include/dataloader.h>
#include "../include/pattern.h"
#include "../include/schedule.h"
#include "../include/common.h"
#include "../include/motif_generator.h"

#include <assert.h>
#include <iostream>
#include <string>
#include <algorithm>

double test_pattern(Graph* g, Pattern &pattern) {
    int thread_num = 16;

    bool is_pattern_valid;
    int performance_modeling_type;
    bool use_in_exclusion_optimize;
    
    performance_modeling_type = 1;
    use_in_exclusion_optimize = false;
    Schedule schedule_our(pattern, is_pattern_valid, performance_modeling_type, 1, use_in_exclusion_optimize, g->v_cnt, g->e_cnt);
    assert(is_pattern_valid==true);

    double t1,t2;
    double total_time = 0;
   
    for(int i = 0; i < 3; ++i) {
        t1 = get_wall_time();
        long long ans_our = g->pattern_matching(schedule_our, thread_num, true);
        t2 = get_wall_time();

        printf("our ans: %lld time: %.6lf\n", ans_our, t2 - t1);
        total_time += (t2 - t1);
        if(i == 2) {
            schedule_our.print_schedule();
        }
        fflush(stdout);
    }
    total_time /= 3;
    printf("%.6lf\n",total_time);
    return total_time;
}

int main(int argc,char *argv[]) {
    Graph *g;
    DataLoader D;

    if(argc != 3) {
        printf("usage: %s graph_file clique_size", argv[0]);
    }

    DataType type = DataType::Patents;

    bool ok = D.load_data(g, type, argv[1]);
    if(!ok) { printf("Load data failed"); return 0; }

    printf("Load data success!\n");
    fflush(stdout);
    int size = atoi(argv[2]);
    Pattern pattern(size);
    for(int i = 0; i < size; i++) {
        for(int j = i + 1; j < size; j++) {
            pattern.add_edge(i, j);
        }
    }

    reduce_edges_for_clique(*g);
    test_pattern(g, pattern);
    delete g;
    return 0;
}
