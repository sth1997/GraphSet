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
    use_in_exclusion_optimize = true;
    Schedule_IEP schedule_our(pattern, is_pattern_valid, performance_modeling_type, 1, use_in_exclusion_optimize, g->v_cnt, g->e_cnt);
    assert(is_pattern_valid==true);

    double t1,t2;
    double total_time = 0;

    int times = 1;
    for(int i = 0; i < times; ++i) {
        t1 = get_wall_time();
        long long ans_our = g->pattern_matching(schedule_our, thread_num);
        t2 = get_wall_time();

        printf("Ans: %lld time: %.6lf\n", ans_our, t2 - t1);
        total_time += (t2 - t1);
        if(i == times - 1) {
            schedule_our.print_schedule();
        }
        fflush(stdout);
    }
    total_time /= 1;
    printf("Counting time cost: %.6lf s\n",total_time);
    return total_time;
}

int main(int argc,char *argv[]) {
    Graph *g;
    DataLoader D;

    if(argc != 4) {
        printf("usage: %s graph_file pattern_size pattern_adj_string\n", argv[0]);
        return 0;
    }

    bool ok = D.fast_load(g, argv[1]);
    // bool ok = D.load_data(g, DataType::Patents, argv[1], false);
    if(!ok) { printf("Load data failed\n"); return 0; }

    printf("Load data success!\n");
    fflush(stdout);
    int pattern_size = atoi(argv[2]);
    const char* pattern_str= argv[3];

    Pattern p(pattern_size, pattern_str);

    test_pattern(g, p);
    delete g;
    return 0;
}