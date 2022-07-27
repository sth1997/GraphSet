#include "../include/task_queue.h"

void Task_Queue::insert(Embedding new_e, int depth)
{
    q[current_depth + 1][graph.get_block_index(new_e.last)].push_back(new_e);
    size[current_depth + 1]++;
    if (size[current_depth + 1] >= Max_size)
    {
        current_depth++;
        current_machine[current_depth] = graph.get_machine_id(); //Todo: 当前机器的编号
    }
}
Embedding Task_Queue::new_task()
{
    int N = graph.get_machine_cnt(); //Todo: 机器数量
    int K = graph.get_machine_id();
    while (current_depth >= 1)
    {
        while (q[current_depth][current_machine[current_depth]].size() == 0 && (current_machine[current_depth] + 1) % N != K)
        {
            (current_machine[current_depth] += 1) %= N;
        }
        if (q[current_depth][current_machine[current_depth]].size())
        {
            size[current_depth]--;
            Embedding e = (*q[current_depth][current_machine[current_depth]].back());
            q[current_depth][current_machine[current_depth]].pop_back();
            return e;
        }
        current_depth--;
    }
    Embedding nul;
    return nul;
}