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

void three_motif_test(Graph* &g) {
    int thread_num = 24;
    Pattern triangle(3, true);
    bool is_pattern_valid;
    Schedule schedule(triangle, is_pattern_valid, 1, 1, 1, g->v_cnt, g->e_cnt, g->tri_cnt);
    assert(is_pattern_valid);

    double t1 = get_wall_time();
    std::pair<long long, long long> ans = g->three_motifs(schedule, thread_num);
    double t2 = get_wall_time();

    printf("triangle count: %lld, chain count %lld\n", ans.first, ans.second);
    printf("3-motifs time %.6lf\n", t2 - t1);
}

void four_motif_test(Graph* &g) {
    int thread_num = 24;
    MotifGenerator MG(4);
    std::vector<Pattern> patterns = MG.generate();
    double total_time = 0;
    for(auto p : patterns) {
        bool is_pattern_valid;
        Schedule schedule(p, is_pattern_valid, 1, 1, 1, g->v_cnt, g->e_cnt, g->tri_cnt);
        assert(is_pattern_valid);
        double t1 = get_wall_time();
        long long ans = g->pattern_matching(schedule, thread_num);
        double t2 = get_wall_time();
        schedule.print_schedule();
        printf("ans %lld, time %.6lf\n", ans, t2 - t1);
        total_time += t2 - t1;
    }
    printf("4-motifs time %.6lf\n", total_time);
}

int main(int argc,char *argv[]) {
    Graph *g;
    DataLoader D;

    const std::string type = argv[1];
    const std::string path = argv[2];
    
    DataType my_type;
    
    GetDataType(my_type, type);

    if(my_type == DataType::Invalid) {
        printf("Dataset not found!\n");
        return 0;
    }

    assert(D.load_data(g,my_type,path.c_str())==true); 

    printf("Load data success!\n");
    fflush(stdout);

//    three_motif_test(g);

    four_motif_test(g);

    delete g;
}

