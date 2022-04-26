#pragma once
#include "../include/graph.h"
#include <cstdio>
#include <vector>
#include <algorithm>
#include <unordered_set>
#include <omp.h>

bool exists_edge(Graph &g, int i, int j){
    // edge[ vertex[i], vertex[i+1]-1 ]

    return true;
    for(int e = g.vertex[i]; e < g.vertex[i+1]; e++) {
        if(g.edge[e] == j) {
            return true;
        }
    }
    return false;
}

void recalculate_max_intersection(Graph &g) {
    printf("old max intersection: %d\n", VertexSet::max_intersection_size);
    int now_max_intersection = 0;
    for(int v = 0; v < g.v_cnt; v++) {
        now_max_intersection = std::max(now_max_intersection, int(g.vertex[v+1] - g.vertex[v]));
    }
    VertexSet::max_intersection_size = now_max_intersection;
    printf("new max_intersection: %d\n", now_max_intersection);
}

void insert_edge( std::vector<std::unordered_set<int>> &to_be_erased, std::vector<int> &degree, int i, int j){
    if(i < j){
        to_be_erased[i].insert(j);
    }
    else {
        to_be_erased[j].insert(i);
    }
}


void erase_edge(Graph &g, std::vector<std::unordered_set<int>> &to_be_erased) {
    int newe = 0;
    printf("start erasing edge\n");
    fflush(stdout);
    for(int v = 0; v < g.v_cnt; v++) {
        int l = newe;
        for(int e = g.vertex[v]; e < g.vertex[v+1]; e++){
            if(to_be_erased[v].count(g.edge[e])) continue;
            g.edge[newe] = g.edge[e];
            newe++; 
        }
        g.vertex[v] = l;
    }
    g.vertex[g.v_cnt] = newe;
    g.e_cnt = newe;
    printf("newe: %d\n", g.e_cnt);
}

void reduce_edges_for_clique(Graph &g) {
    std::vector<int> degree;
    std::vector<std::unordered_set<int>> to_be_erased;
    
    to_be_erased.resize(g.v_cnt);

    printf("trying to reduce edge. the pattern is a clique.\n");

    for(int v = 0; v < g.v_cnt; v++){
        degree.push_back(g.vertex[v+1] - g.vertex[v]);
    }

    for(int v = 0; v < g.v_cnt; v++) {
        for(int e = g.vertex[v]; e < g.vertex[v + 1]; e++) {
            int to_v = g.edge[e];
            if(to_v == v) continue;
            if(!exists_edge(g, to_v, v)){
                printf("not a double edge\n");
                return;
            }
            else {
                if(to_v >= v) {
                    assert(to_v < g.v_cnt && v < g.v_cnt);
                    insert_edge(to_be_erased, degree, v, to_v);
                }
            }
        }
    }

    erase_edge(g, to_be_erased);
    printf("Finish reduce.\n");
}
