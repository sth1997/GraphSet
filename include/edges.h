#pragma once
# include <cstdint>

typedef int32_t v_index_t;
typedef int64_t e_index_t ; 

class Edges 
{
public:
    v_index_t v; // id of vertex
    e_index_t e_cnt; // number of edge
    v_index_t* vet; // number of triangle

    Edges()
    {
        v = 0;
        e_cnt = 0LL;
        vet = nullptr;
    }

    ~Edges() { if(vet != nullptr) delete[] vet; }

    void init(v_index_t x,e_index_t cnt); //将某个点的边列表存进edges
};