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

