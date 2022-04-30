#undef NDEBUG
#include <graph.h>
#include <dataloader.h>
#include <vertex_set.h>
#include <common.h>
#include <schedule.h>

#include <cassert>
#include <cstring>
#include <cstdint>
#include <string>
#include <algorithm>

#include <sys/time.h>
#include <omp.h>

int main(int argc,char *argv[]) {
    LabeledGraph *g;
    DataLoader D;

    const std::string type = argv[1];
    const std::string path = argv[2];
    const int max_edge = atoi(argv[3]);
    const int min_support = atoi(argv[4]);

    DataType my_type;
    
    GetDataType(my_type, type);

    if(my_type == DataType::Invalid) {
        printf("Dataset not found!\n");
        return 0;
    }

    g = new LabeledGraph();
    assert(D.load_labeled_data(g,my_type,path.c_str())==true);
    g->fsm(max_edge, min_support, 1);

    return 0;
}
