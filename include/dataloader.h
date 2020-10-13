#pragma once
#include "graph.h"
#include <unordered_map>
#include <algorithm>
#include <vector>

enum DataType {
    Patents,
    Orkut,
    complete8,
    LiveJournal,
    MiCo,
    Twitter,
    CiteSeer,
    Wiki_Vote,
    Invalid
};

constexpr long long Patents_tri_cnt = 7515023LL;
constexpr long long LiveJournal_tri_cnt = 177820130LL;
constexpr long long MiCo_tri_cnt = 12534960LL;
constexpr long long CiteSeer_tri_cnt = 1166LL;
constexpr long long Wiki_Vote_tri_cnt = 608389LL;
constexpr long long Orkut_tri_cnt = 627584181LL;
constexpr long long Twitter_tri_cnt = 34824916864LL;

class DataLoader {
public:

    bool load_data(Graph* &g, DataType type, const char* path, int oriented_type = 0);
        // pattern_diameter means max distance between two vertex in graph
        // oriented_type is used to reorder dataset
        // oriented_type == 0 do nothing
        //               == 1 high degree first
        //               == 2 low degree first
    bool load_data(Graph* &g, int clique_size);

private:
    static bool cmp_pair(std::pair<int,int>a, std::pair<int,int>b);
    static bool cmp_degree_gt(std::pair<int,int> a,std::pair<int,int> b);
    static bool cmp_degree_lt(std::pair<int,int> a,std::pair<int,int> b);

    long long comb(int n,int k);
    bool general_load_data(Graph* &g, DataType type, const char* path, int oriented_type = 0);
    bool twitter_load_data(Graph* &g, DataType type, const char* path, int oriented_type = 0);

    std::unordered_map<int,int> id;
};
