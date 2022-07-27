#pragma once
# include "schedule_IEP.h"
# include "vertex_set.h"
# include "graph.h"
# include "edges.h"
# include <cstdint>

typedef int32_t v_index_t;
typedef int64_t e_index_t ; 
typedef std::pair<v_index_t,v_index_t> pii ;

class Graph_D
{
public:
    v_index_t v_cnt,block_size; // number of vertex(this part)
    e_index_t e_cnt; // number of edge(this part)
    v_index_t range_l,range_r;

    v_index_t *edge; // edges
    e_index_t *vertex; // v_i's neighbor is in edge[ vertex[i], vertex[i+1]-1]

    Graph_D() 
    {
        v_cnt = 0;
        e_cnt = 0LL;
        edge = nullptr;
        vertex = nullptr;
    }

    ~Graph_D() 
    {
        if(edge != nullptr) delete[] edge;
        if(vertex != nullptr) delete[] vertex;
    }

    void init(Graph* graph);
    bool in_this_part(v_index_t x);
    void get_neighbor(v_index_t x,Edges& E); //获取一个在此机器的点的信息
    void ask_neighbor(v_index_t x,Edges& E); //获取一个不在此机器的点的信息
    void give_neighbor(); //回复其他机器的询问
};