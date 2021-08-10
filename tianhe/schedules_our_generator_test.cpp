// test all schedules which inner k loops are not connected with each other
// and use our restricts

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
#include <chrono>

int node_cnt = 1;
int my_rank;

void test_pattern(Graph* g, Pattern &pattern) {
    int work_cnt = 0;

    int thread_num = 24;
    double t1,t2;

    bool is_pattern_valid;
    int performance_modeling_type;
    bool use_in_exclusion_optimize;

    t1 = get_wall_time();
    Schedule_IEP schedule_our(pattern, is_pattern_valid, 1, 1, true, g->v_cnt, g->e_cnt, g->tri_cnt);
    assert(is_pattern_valid);
    t2 = get_wall_time();

    printf("Init time : %.6lf\n", t2 - t1);

    int k_val = schedule_our.get_k_val();
    printf("k_val : %d\n", k_val);

    t1 = get_wall_time();
    long long ans = g->pattern_matching(schedule_our, thread_num);
    t2 = get_wall_time();
    printf("our time : %.6lf\n", t2 - t1);

    int size = pattern.get_size();
    const int* adj_mat = pattern.get_adj_mat_ptr();
    int rank[size];
    for(int i = 0; i < size; ++i) rank[i] = i;
    do{
        Pattern cur_pattern(size);
        for(int i = 0; i < size; ++i)
            for(int j = 0; j < i; ++j)
                if( adj_mat[INDEX(i,j,size)])
                    cur_pattern.add_edge(rank[i],rank[j]);

        bool valid = true;
        const int* cur_adj_mat = cur_pattern.get_adj_mat_ptr();
        for(int i = 1; i < size; ++i) {
            bool have_edge = false;
            for(int j = 0; j < i; ++j)
                if(cur_adj_mat[INDEX(i,j,size)]) {
                    have_edge = true;
                    break;
                }
            if(!have_edge) {
                valid = false;
                break;
            }
        }
        if(!valid) continue;

        int my_k = 1;
        for(int k = 2; k <= size; ++k) {
            bool flag = true;
            for(int i = size - k + 1; i < size; ++i)
                if(cur_adj_mat[INDEX(size - k,i,size)]) {
                    flag = false;
                    break;
                }
            if(flag == false) {
                my_k = k - 1;
                break;
            }
        }
        assert(my_k <= k_val);
        if(my_k != k_val) continue;

        Schedule_IEP schedule(cur_pattern, is_pattern_valid, 0, 1, false, g->v_cnt, g->e_cnt, g->tri_cnt);
        if(is_pattern_valid == false) continue;

        ++work_cnt;
        if( work_cnt % node_cnt != my_rank) continue;

        for(int i = 0; i < size; ++i,puts(""))
            for(int j = 0; j < size; ++j)
                printf("%d",cur_adj_mat[INDEX(i,j,size)]);
        puts("");

        t1 = get_wall_time();
        long long ans2 = g->pattern_matching(schedule, thread_num);
        t2 = get_wall_time();

        if( ans2 > 0 ) {
            assert(ans == ans2);
            printf("time : %.6lf\n", t2 - t1);
        }
        else printf("timeout\n");
        fflush(stdout);
    } while( std::next_permutation(rank, rank + size));
    
}

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
    
    int size = atoi(argv[2]);
    char* adj_mat = argv[3];
    my_rank = atoi(argv[4]);

    Pattern p(size, adj_mat);
    test_pattern(g, p);
    
    delete g;
}
