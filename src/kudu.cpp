#include "../include/graph.h"
#include "../include/graph_d.h"
#include "../include/embedding.h"

//1号线程用于通信
void fetch(std::queue<Embedding> &que)
{
    std::queue<Embedding> new_que;
    Edges edge;
    while (! que.empty())
    {
        Embedding now = que.front();
        que.pop();
        if (now.get_state() == 0)
        {
            if (in_this_part(now.get_request()))
            {
                get_neighbor(now.get_request(), edge);
            }
            else
            {
                ask_neighbor(now.get_request(), edge);
            }
            now.add_edge(edge);
            new_que.push(now);
        }
        else
        {
            new_que.push(now);
        }
    }//Todo：一组同时进行通信
    que = new_que;
}

int main(int argc, char *argv[])
{
    //调用Graph_D::init()
    //0号线程用来发送信息，其他进入kudu函数
}