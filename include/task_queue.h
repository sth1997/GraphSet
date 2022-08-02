#pragma once
#include "graph_d.h"
#include "embedding.h"
#include <cstring>
#include <vector>
#include <algorithm>

const int pattern_size = 3 + 1;//模式的大小
const int machine_cnt = 4;//机器的数量

class Task_Queue
{
public:
    Task_Queue(Graph_D *G) //G表示分布式图储存
    {
        graph = G;
        current_depth = 1;
        current_machine = new int[pattern_size];
        size = new int[pattern_size];
        index = new int[pattern_size][machine_cnt];
        is_commued = new int[pattern_size][machine_cnt];
        commu = new int[pattern_size];
        memset(current_machine, 0, sizeof(current_machine));
        memset(size, 0, sizeof(size));
        memset(commu, 0, sizeof(commu));
        for (int i = 0; i < pattern_size; i++)
        {
            for (int j = 0; j < machine_cnt; j++)
            {
                index[i][j] = 0;
                is_commued[i][j] = 0;
            }
        }
    }
    ~Task_Queue()
    {
        delete[] current_machine;
        delete[] size;
        delete[] commu;
        for (int i = 0; i < pattern_size; i++)
        {
            delete[] index[i];
            delete[] is_commued[i];
        }
        delete[] index;
        delete[] is_commued;
    }
    const int Max_size = 100000;//每层达到Max_size后就将此层设为当前层
    int current_depth;//当前正在扩展哪一层
    int* current_machine;//每层当前在哪台机器
    int* size;//记录每层目前大小
    std::vector<Embedding> q[pattern_size][machine_cnt]; //每一层，每一台机器，开一个vector储存
    int (*index)[machine_cnt]; //每一层，每一台机器，已经算到了第几个
    int (*is_commued)[machine_cnt]; //每一层，每一台机器，是否已经通信完了
    int *commu; //每一层，已经通信到了第几个
    void insert(Embedding new_e, bool is_root = false);//加入一个embedding
    Embedding* new_task();//获取一个新任务,层数与embedding的size相同
    Graph_D *graph;
    Embedding nul;
};