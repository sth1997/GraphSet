#pragma once
# include <cstdint>
# include <vector>
# include "embedding.h"
# include "edges.h"

typedef int32_t v_index_t ;
typedef int64_t e_index_t ; 

const int buffer_size=100000; //循环队列的大小

class Comm
{
private:
    v_index_t* askx;
    Embedding** aske,anse;
    Edges* ans; //用完记得将vet数组delete（只用于暂时存放通信结果）
    MPI_Request* recv_request;

    int l_ask,l_ans,r_ask,r_ans,size;
    bool all_solved;
    e_index_t max_degree;
    Graph_D* graph;

    Comm() 
    {
        l_ask=l_ans=r_ask=r_ans=0;
        all_solved=0;
        size=buffer_size;
        askx=new v_index_t[size+2];
        aske=new Embedding*[size+2];
        anse=new Embedding*[size+2];
        ans=new Edges[size+2];
        recv_request=new MPI_Request[size+2];
    }

    ~Comm() {}

public:
    void ins_ask(v_index_t x,Embedding* e); //增加一个询问
    void comm_send(); //线程0
    void comm_recv(); //线程1
    void computation_done();
    void set_max_degree(e_index_t s);
};