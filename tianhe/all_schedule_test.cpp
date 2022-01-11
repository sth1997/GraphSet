#include <../include/graph.h>
#include <../include/dataloader.h>
#include "../include/pattern.h"
#include "../include/schedule_IEP.h"
#include "../include/common.h"
#include "../include/motif_generator.h"

#include <assert.h>
#include <iostream>
#include <string>
#include <algorithm>

void test_pattern(Graph* g, Pattern &pattern) {

    int thread_num = 16;
    double t1,t2,t3,t4;

    bool is_pattern_valid;
    int performance_modeling_type;
    bool use_in_exclusion_optimize;

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

        Schedule_IEP schedule_our(cur_pattern, is_pattern_valid, 0, 1, true, g->v_cnt, g->e_cnt, g->tri_cnt);
        
        if(is_pattern_valid == false) continue;
        
        t1 = get_wall_time();
        long long ans = g->pattern_matching(schedule_our, thread_num);
        t2 = get_wall_time();
        
        printf("ans %lld time: %.6lf\n", ans, t2 - t1);
        schedule_our.print_schedule();
        
        fflush(stdout);

    } while( std::next_permutation(rank, rank + size));
    
}

int main(int argc,char *argv[]) {
    Graph *g;
    DataLoader D;

    bool ok;
    ok = D.fast_load(g, argv[1]);

    if (!ok) {
        printf("data load failure :-(\n");
        return 0;
    }

    int size = atoi(argv[2]);
    char* adj_mat = argv[3];
    
    Pattern p(size, adj_mat);
    test_pattern(g, p);
    
    delete g;
}
