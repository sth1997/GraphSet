#pragma once
#include "edges.h"

class Embedding //维护嵌入的点集、活动边列表
{
public:
    Embedding()
    {
        size = 0;
        state = 1;
        last = -1;
        list = nullptr;
        father = nullptr;
    }
    Embedding(Embedding *fa, int v) //在它的父亲Extendable Embedding中增加一个新的点，即伪代码中的create_extendable_embedding，状态为Pending
    {
        father = fa;
        size = (fa->size) + 1;
        state = 0;
        last = v;
        list = new Edges*[size];
        for (int i = 0; i < size - 1; i++)
        {
            list[i] = fa->list[i];
        }
    }
    ~Embedding()
    {
        for (int i = 0; i < size; i++)
        {
            if (list[i] != nullptr)
            {
                delete[] list[i];
            }
        }
        if (list != nullptr) delete[] list;
        father = nullptr;
        size = 0;
        state = 3;
        last = 0;
    }
    inline int get_state();
    inline int get_size();
    inline int set_state(int st);
    inline int get_request(); //若当前embedding为Pending状态，则返回需要访问的节点编号，否则返回-1
    void add_edge(Edges edge); //传入需要的边列表，将状态变为Ready
    inline Embedding* get_father();
    inline Edges **get_list(); //返回活动边列表的指针数组
    Edges* get_edge(int u); //返回u节点的所有边，不存在则返回nullptr
    //Edges get_union_list(int *vet); //Todo 返回一个点集的公共邻点列表，使用Vertical computation sharing优化（是否可行未知
private:
    int state; //state = 0, 1, 2, 3 分别表示Pending, Ready, Zombie, Terminated
    int size;
    int last;
    Edges **list;
    Embedding *father;
};
