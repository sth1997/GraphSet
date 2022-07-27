#include "../include/graph_d.h"
#include "../include/graph.h"
#include <mpi.h>
#include <omp.h>

void Graph_D::init(Graph* graph)
{
    Graph G=graph;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Barrier(MPI_COMM_WORLD);
    block_size=(G.v_cnt + comm_sz - 1) / comm_sz;
    range_l=block_size*my_rank;
    range_r=std::min(block_size*(my_rank+1),G.v_cnt);
    v_cnt=0,e_cnt=0;
    for (v_index_t i=range_l;i<=range_r;++i)
    {
        vertex[v_cnt]=G.vertex[i];
        for (e_index_t j=G.vertex[i];j<G.vertex[i+1];++j)
            edge[++e_cnt]=G.edge[j];
        v_cnt++;
    }
}

bool in_this_part(v_index_t x)
{
    return (range_l<=x)&&(x<=range_r);
}

void get_neighbor(v_index_t x,Edges& E) //获取一个在此机器的点的信息
{
    E.v=x;
    x=x-range_l;
    E.e_cnt=vertex[x+1]-vertex[x];
    E.vet=edge[ vertex[x] ];
}

void ask_neighbor(v_index_t x,Edges& E); //获取一个不在此机器的点的信息
{
    give_neighbor(); //因为现在是用阻塞通信，所以先把询问处理完防止死锁，之后会改
    int tar=x/block_size; //x所存的机器位置
    MPI_Request rq_recv;
    MPI_Status status;
    bool flag;
    MPI_send(x,1,MPI_INT,tar,0,MPI_COMM_WORLD);
    E.v=x;
    MPI_IRecv(E.e_cnt,1,MPI_INT,tar,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE,rq_recv);
    MPI_Test(&rq_recv, &flag, &status);
    if(flag)
        MPI_Recv(E.vet,E.e_cnt,MPI_INT,tar,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        //https://mpitutorial.com/tutorials/dynamic-receiving-with-mpi-probe-and-mpi-status/zh_cn/
        //这里有一个关于不定长度的数据传输，有空改一下（没必要接收两次）
}

void give_neighbor();
{
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    for (int i=0;i<comm_sz;++i)
    {
        if(i==my_rank) continue;
        MPI_Request rq_recv;
        MPI_Status status;
        bool flag;
        int ask;
        MPI_IRecv(ask,1,MPI_INT,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE,rq_recv);
        MPI_Test(&rq_recv, &flag, &status);
        if(flag)
        {
            Edges E;
            get_neighbor(ask,E);
            MPI_Send(E.e_cnt,1,MPI_INT,i,0,MPI_COMM_WORLD);
            MPI_Send(E.vet.E.e_cnt,MPI_INT,i,0,MPI_COMM_WORLD);
        }
    }
}