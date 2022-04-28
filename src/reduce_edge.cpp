#include "../include/graph.h"
#include <cstdio>
#include <vector>
#include <algorithm>
#include <unordered_set>
#include <omp.h>

void erase_edge(Graph &g) {
    int newe = 0;
    printf("start erasing edge\n");
    fflush(stdout);
    for(int v = 0; v < g.v_cnt; v++) {
        int l = newe;
        for(long long e = g.vertex[v]; e < g.vertex[v+1] && e < g.e_cnt; e++){
            if(g.edge[e] >= v) continue;
            g.edge[newe] = g.edge[e];
            newe++; 
        }
        g.vertex[v] = l;
    }
    g.vertex[g.v_cnt] = newe;
    g.e_cnt = newe;
    printf("newe: %lld\n", g.e_cnt);
}

void reduce_edges_for_clique(Graph &g) {

    printf("trying to reduce edge. the pattern is a clique.\n");

    erase_edge(g);
    printf("Finish reduce.\n");
}
