#include "graph.h"
#include "dataloader.h"
#include "schedule.h"
#include "pattern.h"

#include <cstdio>
#include <cstdlib>
#include <chrono>

int main(int argc, char* argv[])
{
    Graph *g;
    DataLoader D;

    bool ok = D.fast_load(g, argv[1]);
    if (!ok) {
        printf("unable to load data\n");
        return 0;
    }

    int n = atoi(argv[2]);
    const char *adj_mat = argv[3];
    bool pattern_valid;
    Pattern p(n, adj_mat);
    Schedule schedule(p, pattern_valid, 1, 1, false, g->v_cnt, g->e_cnt, g->tri_cnt);

    using namespace std::chrono;
    auto t1 = system_clock::now();
    int support = g->calculate_support(schedule);
    auto t2 = system_clock::now();

    auto elapsed = duration_cast<microseconds>(t2 - t1).count() * 1e-6;
    printf("support = %d (%g seconds)\n", support, elapsed);

    return 0;
}
