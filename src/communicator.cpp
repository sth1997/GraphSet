#include "../include/communicator.h"
#include "../include/graph_d.h"
#include "../include/task_queue.h"
#include <mpi.h>
#include <omp.h>

void Comm::give_ans() //线程0-回复其他机器的询问
{
    int comm_sz,my_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    v_index_t* ask;
    MPI_Request* recv_r;
    Edges* e;
    MPI_Status status;
    int flag=0;
    ask=new v_index_t[comm_sz+1];
    recv_r=new MPI_Request[comm_sz+1];
    e=new Edges[comm_sz+1];
    for (int i=0;i<comm_sz;++i)
    {
        if(i==my_rank) continue;
        MPI_Irecv(&ask[i],1,MPI_INT,i,MPI_ANY_TAG,MPI_COMM_WORLD,&recv_r[i]);
    }
    while(!all_solved)
    {
        for (int i=0;i<comm_sz;++i)
        {
            if(i==my_rank) continue;
            MPI_Test(&recv_r[i],&flag,&status);
            if(flag==0) continue;
            graph->get_neighbor(ask[i],e[i]);
            MPI_Send(e[i].vet,e[i].e_cnt,MPI_INT,i,ask[i],MPI_COMM_WORLD);
            //TODO：改为Isend，但要检查一下上次的有没有发出
        }
    }
}

void Comm::ask_ans(Task_Queue* task)//线程1
{
    int comm_sz, my_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    while (!all_solved)
    {
        #pragma omp flush(task)
        int depth = task->current_depth;
        int index = task->commu[depth];
        (task->commu[depth] += 1) %= comm_sz;
        if (depth == 0)
        {
            break;
        }
        std::vector<Embedding> vec=task->q[depth][index];
        Edges edge;
        if (! task.is_commued[depth][index])
        {
            if(index==my_rank)
            {
                int x;
                for (int i=0;i<(int)vec.size();++i)
                {
//                    if(vec[i].get_state()!=0) break; //Todo:加一个表示这组是否通信完成的标识符
                    x=vec[i].get_request();
                    graph->get_neighbor(x,edge);
                    vec[i].add_edge(edge);
                }
            }
            else
            {
                int size=vec.size();
                MPI_Status status;
                for (int i=0;i<size;++i)
                {
//                    if(vec[i].get_state()!=0) break;
                    int x=vec[i].get_request();
                    MPI_Send(&x,1,MPI_INT,index,x,MPI_COMM_WORLD);
                }
                for (int i=0;i<size;++i)
                {
//                    if(vec[i].get_state()!=0) break;
                    MPI_Recv(buffer,max_degree,MPI_INT,index,vec[i].get_request(),MPI_COMM_WORLD,&status);
                    Edges edge;
                    edge.v=vec[i].get_request();
                    int cnt=0;
                    MPI_Get_count(&status,MPI_INT,&cnt);
                    edge.e_cnt=cnt;
                    edge.vet=new v_index_t[edge.e_cnt];
                    for (int j=0;j<edge.e_cnt;++j)
                        edge.vet[j]=buffer[j];
                    vec[i].add_edge(edge);
                }
            }
            task.is_commued[depth][index] = 1;
        }
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

/*
void Comm::comm_recv() //线程1
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
    while(!all_task_solved)
    {
        if(l_ans==r_ans) continue;
        int pos=l_ans;
        MPI_Status status;
        MPI_Wait(&recv_request[pos],&status);
        anse[pos]->add_edge(ans[r_ans]);
        anse[pos]->set_state(1);
        l_ans=(l_ans+1)%size;
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
        }
*/
