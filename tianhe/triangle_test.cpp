#include <../include/graph.h>
#include <../include/dataloader.h>
#include "../include/pattern.h"
#include "../include/schedule.h"
#include "../include/common.h"
#include "../include/motif_generator.h"
#include "../include/vertex_set.h"

#include <assert.h>
#include <iostream>
#include <string>
#include <algorithm>

void test_pattern(Graph* g, const Pattern &pattern, int performance_modeling_type, int restricts_type, bool use_in_exclusion_optimize = false) {
    printf("max intersection size %d\n", VertexSet::max_intersection_size);
    int thread_num = 24;
    double t1,t2;
    
    bool is_pattern_valid;
    Schedule schedule(pattern, is_pattern_valid, 1, 1, use_in_exclusion_optimize, g->v_cnt, g->e_cnt, g->tri_cnt);
    assert(is_pattern_valid);

    t1 = get_wall_time();
    long long ans = g->pattern_matching(schedule, thread_num);
    t2 = get_wall_time();

    printf("ans %lld\n", ans);
    printf("time %.6lf\n", t2 - t1);
//    printf("intersection %lld, %lld", g->intersection_times_low, g->intersection_times_high);
//    printf("depth_cnt %lld %lld %lld\n", g->dep1_cnt, g->dep2_cnt, g->dep3_cnt);
    schedule.print_schedule();
    const auto& pairs = schedule.restrict_pair;
    printf("%d ",pairs.size());
    for(auto& p : pairs)
        printf("(%d,%d)",p.first,p.second);
    puts("");
    fflush(stdout);

}

int main(int argc,char *argv[]) {
    Graph *g;
    DataLoader D;

    const std::string type = argv[1];
    const std::string path = argv[2];
    
//    int size = atoi(argv[3]);
//    char* adj_mat = argv[4];

//    int test_type = atoi(argv[5]);
    
    DataType my_type;
    
    GetDataType(my_type, type);

    if(my_type == DataType::Invalid) {
        printf("Dataset not found!\n");
        return 0;
    }

    assert(D.load_data(g,my_type,path.c_str())==true); 
    //assert(D.load_data(g,100)==true); 

    printf("Load data success!\n");
    fflush(stdout);

    Pattern p(3, true);
    test_pattern(g, p, 1, 1, true);

/*
    Pattern p1(PatternType::sigmod2020_guo_q1);
    test_pattern(g, p1, 1, 1, true);
    
    Pattern p2(PatternType::sigmod2020_guo_q2);
    test_pattern(g, p2, 1, 1, true);
    
    Pattern p3(PatternType::sigmod2020_guo_q3);
    test_pattern(g, p3, 1, 1, true);
    
    Pattern p4(PatternType::sigmod2020_guo_q4);
    test_pattern(g, p4, 1, 1, true);
    
    Pattern p5(PatternType::sigmod2020_guo_q5);
    test_pattern(g, p5, 1, 1, true);
    
    Pattern p6(PatternType::sigmod2020_guo_q6);
    test_pattern(g, p6, 1, 1, true);
*/
    delete g;
}
