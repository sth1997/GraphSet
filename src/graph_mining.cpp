#include "../include/graph.h"
#include "../include/graph_d.h"
#include "../include/embedding.h"
#include <queue>
#include <vector>
#include <algorithm>
#include <omp.h>

//1号线程用于通信
void communication(std::vector<Embedding> &vec)
{
    Edges edge;
    for (int i = 0; i < (int)vec.size(); i++)
    {
        if (vec[i].get_state() == 0)
        {
            if (graph.in_this_part(vec[i].get_request()))
            {
                graph.get_neighbor(vec[i].get_request(), edge);
            }
            else
            {
                graph.ask_neighbor(vec[i].get_request(), edge);
            }
            vec[i].add_edge(edge);
        }
    }//Todo：一组同时进行通信
}

void computation(std::vector<Embedding> (*extend)(Embedding *e), Embedding *e)
{
    std::vector<Embedding> vec = (*extend)(e);
    for (int i = 0; i < (int)vec.size(); i++)
    {
        #pragma omp flush(task)
        task.insert(vec[i]);
    }
    (*e).set_state(2);
}

std::vector<Embedding> triangle_extend(Embedding *e)
{
    std::vector<Embedding> vec;
    if ((*e).get_size() == 1)
    {
        Edges **list = (*e).get_list();
        int cnt = list[0]->e_cnt;
        for (int i = 0; i < cnt; i++)
        {
            Embedding ep(e, list[0]->vet[i]);
            vec.push_back(ep);
        }
    }
    else
    {
        Edges **list = (*e).get_list();
        int cnt1 = list[0]->e_cnt;
        int cnt2 = list[1]->e_cnt;
        for (int i = 0; i < cnt1; i++)
        {
            bool flag = false;
            for (int j = 0; j < cnt2; j++)
            {
                if ((list[0]->vet[i]) == (list[1]->vet[j]))
                {
                    flag = true;
                }
            }
            if (flag)
            {
                Embedding ep(e, list[0]->vet[i]);
                vec.push_back(ep);
            }
        }
    }
    return vec;
}

void graph_mining(std::vector<Embedding> (*extend)(Embedding *e))
{
    Task_Queue task(graph);
    Embedding nul;
    for (int i = graph.range_l; i < graph.range_r; i++) //加入一个点的embedding
    {
        task.insert((Embedding)(nul, i), true);
    }
    #pragma omp parallel shared(task)
    {
        int my_rank = omp_get_thread_num();
        int thread_count = omp_get_num_threads();
        if (my_rank == 0)
        {
            //发送数据
        }
        if (my_rank == 1)
        {
            while (true)
            {
                #pragma omp flush(task)
                int depth = task.current_depth, index = task.current_machine[task.current_depth];
                if (depth == 0)
                {
                    break;
                }
                communication(task.q[depth][index]);
            }
        }
        if (my_rank > 1)
        {
            while (true)
            {
                #pragma omp flush(task)
                Embedding* e = task.new_task();
                if ((*e).size() == 0)
                {
                    break;
                }
                computation(func, e);
            }
        }
    }
    //Todo: 向其他机器发送结束信号
}