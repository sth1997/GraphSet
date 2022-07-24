#include "../include/embedding.h"

inline int Embedding::getsize()
{
    return size;
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