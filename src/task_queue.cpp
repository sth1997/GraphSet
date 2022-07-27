#include "../include/graph_d.h"
#include "../include/embeddiing.h"
#include "../include/task_queue.h"
#include <omp.h>

//Todo: 多线程
void Task_Queue::insert(Embedding new_e, bool is_root = false)
{
    q[current_depth + 1][(*graph).get_block_index(new_e.last)].push_back(new_e);
    size[current_depth + 1]++;
    if (is_root && size[current_depth + 1] >= Max_size)
    {
        current_depth++;
        int N = (*graph).get_machine_cnt(); //Todo: 机器数量
        int K = (*graph).get_machine_id(); //Todo: 当前机器的编号
        current_machine[current_depth] = K;
        for (int i = 0; i < N; i++)
        {
            index[current_depth][i] = 0;
        }
    }
}

//Todo: 多线程
Embedding* Task_Queue::new_task()
{
    int N = (*graph).get_machine_cnt(); //Todo: 机器数量
    int K = (*graph).get_machine_id();
    while (current_depth >= 1)
    {
        while (index[current_depth][current_machine[current_depth]] == (int)q[current_depth][current_machine[current_depth]].size() && (current_machine[current_depth] + 1) % N != K)
        {
            (current_machine[current_depth] += 1) %= N;
        }
        if (index[current_depth][current_machine[current_depth]] < (int)q[current_depth][current_machine[current_depth]].size())
        {
            while (q[current_depth][current_machine[current_depth]][index[current_depth][current_machine[current_depth]]].get_state() != 1);
            return &q[current_depth][current_machine[current_depth]][index[current_depth][current_machine[current_depth]]];
        }
        for (int i = 0; i < N; i++)
        {
            q[current_depth][i].clear();
        }
        current_depth--;
    }
    Embedding nul;
    return nul;
}