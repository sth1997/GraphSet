#include "../include/edges.h"
#include "../include/embedding.h"

int Embedding::get_state()
{
    return state;
}

int Embedding::get_size()
{
    return size;
}

void Embedding::set_state(int st)
{
    state = st;
}

int Embedding::get_request()
{
    return last;
}

void Embedding::add_edge(Edges edge)
{
    list[size - 1] = new Edges(edge);
    state = 1;
}

Embedding* Embedding::get_father()
{
    return father;
}

Edges** Embedding::get_list()
{
    return list;
}

Edges* Embedding::get_edge(int u)
{
    for (int i = 0; i < size; i++)
    {
        if ((list[i]->v) == u)
        {
            return list[i];
        }
    }
    return nullptr;
}