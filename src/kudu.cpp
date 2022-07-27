#include "../include/graph.h"
#include "../include/graph_d.h"
#include "../include/embedding.h"
#include <queue>
#include <vector>
#include <algorithm>

//主线程用于通信
void fetch(std::vector<Embedding> &vec)
{
    Edges edge;
    for (int i = 0; i < (int)vec.size(); i++)
    {
        if (vec[i].get_state() == 0)
        {
            if (in_this_part(vec[i].get_request()))
            {
                get_neighbor(vec[i].get_request(), edge);
            }
            else
            {
                ask_neighbor(vec[i].get_request(), edge);
            }
            vec[i].add_edge(edge);
        }
    }//Todo：一组同时进行通信
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
            int vet = list[0]->vet[i];
            Embedding ep(e, vet[i]);
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
                Embedding ep(e, vet[i]);
                vec.push_back(ep);
            }
        }
    }
    return vec;
}
