#include "../include/edges.h"
#include "../include/embedding.h"

inline int Embedding::get_state()
{
    return state;
}

inline int Embedding::get_size()
{
    return size;
}

inline void Embedding::set_state(int st)
{
    state = st;
}

inline int Embedding::get_request()
{
    return last;
}

void Embedding::add_edge(Edges edge)
{
    list[size - 1] = new Edges(edge);
    state = 1;
}

inline Embedding* Embedding::get_father()
{
    return father;
}

inline Edges** Embedding::get_list()
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