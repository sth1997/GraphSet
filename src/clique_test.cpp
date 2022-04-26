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

void test_pattern(Graph* g, Pattern &pattern) {
    int thread_num = 64;
    int tri_cnt = 627584181;

    double t1,t2,t3,t4;
    
    bool is_pattern_valid;
    int performance_modeling_type;
    bool use_in_exclusion_optimize;
    
    performance_modeling_type = 1;
    use_in_exclusion_optimize = true;
    Schedule schedule_our(pattern, is_pattern_valid, performance_modeling_type, use_in_exclusion_optimize, g->v_cnt, g->e_cnt, tri_cnt);
    assert(is_pattern_valid==true);


    // std::vector< std::vector< std::pair<int,int> > >restricts;
    // schedule_our.restricts_generate(schedule_our.get_adj_mat_ptr(), restricts);

    // std::vector< std::pair<int,int> > our_pairs;
    // schedule_our.restrict_selection(g->v_cnt, g->e_cnt, tri_cnt, restricts, our_pairs);
    // schedule_our.add_restrict(our_pairs);
   
    for(int i = 0; i < 3; ++i) {
        t1 = get_wall_time();
        long long ans_our = g->pattern_matching(schedule_our, thread_num, true);
        t2 = get_wall_time();

        printf("our ans: %lld time: %.6lf\n", ans_our, t2 - t1);
        if(i == 2) {
            schedule_our.print_schedule();
            // for(int i = 0; i < our_pairs.size(); ++i)
            //     printf("(%d,%d)",our_pairs[i].first, our_pairs[i].second);
            // puts("");
        }
        fflush(stdout);
    }

}

int main(int argc,char *argv[]) {
    Graph *g;
    DataLoader D;

    std::string type = "Orkut";
    std::string path = "/home/cqq/data/orkut.g";
    DataType my_type;
    if(type == "Orkut") my_type = DataType::Orkut;
    else {
        printf("invalid DataType!\n");
    }

    bool ok = D.fast_load(g, path.c_str()); 

    

    fflush(stdout);

    if(!ok){
        return 0;
    }

    printf("Load data success!\n");
    fflush(stdout);
    Pattern pattern(4);
    pattern.add_edge(0, 1);
    pattern.add_edge(0, 2);
    pattern.add_edge(0, 3);
    pattern.add_edge(1, 2);
    pattern.add_edge(1, 3);
    pattern.add_edge(2, 3);

    reduce_edges_for_clique(*g);

    test_pattern(g, pattern);
    /*
       test_pattern(g, PatternType::Rectangle);
       test_pattern(g, PatternType::QG3);
       test_pattern(g, PatternType::Pentagon);
       test_pattern(g, PatternType::House);
       test_pattern(g, PatternType::Hourglass);
       test_pattern(g, PatternType::Cycle_6_Tri);
       test_pattern(g, PatternType::Clique_7_Minus);
     */
/*
    for(int size = 3; size < 7; ++size) {
        MotifGenerator mg(size);
        std::vector<Pattern> patterns = mg.generate();
        for(Pattern& p : patterns) {
            test_pattern(g, p);
        }
    }
*/
    delete g;
}
