#pragma once
# include "schedule_IEP.h"
# include "vertex_set.h"
# include "edges.h"
# include <cstdint>

typedef int32_t v_index_t;
typedef int64_t e_index_t ; 

class Graph_D 
{
    public:
        v_index_t v_cnt; // number of vertex
        e_index_t e_cnt; // number of edge
        long long tri_cnt; // number of triangle

}