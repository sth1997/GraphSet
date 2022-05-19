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

int main(int argc,char *argv[]) {
    Graph *g;
    DataLoader D;

    if(argc != 3) {
        printf("usage: %s graph_file pattern_size\n", argv[0]);
        return 0;
    }
    
    bool ok = D.fast_load(g, argv[1]);
    if(!ok) { printf("Load data failed\n"); return 0; }

    printf("Load data success!\n");
    fflush(stdout);
    int size = atoi(argv[2]);

    int thread_count = 16;
    g->motif_counting(size, thread_count);
    g->motif_counting_3(thread_count);
    delete g;
    return 0;
}

