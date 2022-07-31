#include "../include/communicator.h"
#include "../include/graph_d.h"
#include <mpi.h>
#include <omp.h>

void Comm::ins_ask(v_index_t x,Embedding* e) //增加一个询问
{
    assert( (r_ask+1)%size==l_ask && "等待询问队列长度不足" );
    r_ask=(r_ask+1)%size;
    askx[r_ask]=x;
    aske[r_ask]=e; 
}

void Comm::comm_send() //线程0
{
    while(!all_task_solved)
    {
        if(l_ask==r_ask) continue;
        int pos=(l_ask+1)%size,tar;
        tar=graph->get_block_index(asks[pos]);
        v_index_t now=asks[pos];
        MPI_Isend(&asks[pos],1,MPI_INT,tar,0,MPI_COMM_WORLD);
        assert( (r_ans+1)%size==l_ans && "等待回答队列长度不足" );
        r_ans=(r_ans+1)%size;
        ans[r_ans].v=asks[pos];
        anse[r_ans]=aske[pos];
        ans[r_ans].vet=new v_index_t[max_degree];
        MPI_Irecv(ans[r_ans].vet,max_degree,MPI_INT,tar,asks[pos],MPI_COMM_WORLD,recv_request[r_ans]);
        l_ask=(l_ask+1)%size;
    }
}

void Comm::comm_recv() //线程1
{
    while(!all_task_solved)
    {
        if(l_ans==r_ans) continue;
        int pos=l_ans;
        MPI_Status status;
        MPI_Wait(&recv_request[pos],&status);
        anse[pos]->add_edge(ans[r_ans]);
        anse[pos]->set_state(1);
        l_ans=(l_ans+1)%size;
        /*
        int nl=l_ans,nr=r_ans; //在等待过程中可能会出现新询问，所以记录一下当前的lr
        MPI_Status status;
        int flag=0; 
        for (int i=nl;i<=nr;++i)
        {
            MPI_Test(&recv_request[i],&flag,&status);
            if(flag)
            {
                if(i==l_ans) l_ans=(l_ans+1)%size;
            }
        }*/

    }
}

void Comm::computation_done()
{
    all_solved=true;
}

void Comm::set_max_degree(e_index_t s)
{
    max_degree=s;
}