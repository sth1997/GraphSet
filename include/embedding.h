class Embedding //维护嵌入的点集、活动边列表
{
public:
    Embedding();
    {
        size = 0;
        list = nullptr;
        father = nullptr;
    }
    Embedding(Embedding *fa, Edges edge); //在它的父亲Extendable Embedding中增加一个新的点，即伪代码中的create_extendable_embedding
    {
        father = fa;
        size = (fa->size) + 1;
        list = new (*Edges)[size];
        for (int i = 0; i < size - 1; i++)
        {
            list[i] = fa->list[i];
        }
        list[size - 1] = new Edges(edge);
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
    }
    inline int get_size();
    inline Embedding* get_father();
    inline Edges **get_list(); //返回活动边列表的指针数组
    Edges* get_edge(int u); //返回u节点的所有边，不存在则返回nullptr
    //Edges get_union_list(int *vet); //Todo 返回一个点集的公共邻点列表，使用Vertical computation sharing优化（是否可行未知
private:
    int size;
    Edges **list;
    Embedding *father;
}
