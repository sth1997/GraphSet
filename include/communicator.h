#pragma once
# include <cstdint>
# include <vector>
# include "embedding.h"
# include "edges.h"
# include "task_queue.h"

typedef int32_t v_index_t ;
typedef int64_t e_index_t ; 

const int buffer_size=100000;
//Todo:先获取maxdegree再生成Comm

class Comm
{
public:
    v_index_t* buffer;
    bool all_solved;
    e_index_t max_degree;
    Graph_D* graph;

    Comm() 
    {
        all_solved=0;
        buffer=new v_index_t[buffer_size];
    }

    ~Comm() 
    {
        if(buffer != nullptr) delete[] buffer;
    }

    void give_ans(); //线程0
    void ask_ans(Task_Queue* task); //线程1
    void computation_done();
    void set_max_degree(e_index_t s);
};