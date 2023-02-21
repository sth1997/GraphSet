#pragma once
#include "graph.h"
#include "labeled_graph.h"
#include <unordered_map>
#include <algorithm>
#include <vector>
#include <cstdint>
#include <bitset>
#include <cassert>
#include <cstring>

enum DataType {
    Patents,
    Orkut,
    complete8,
    LiveJournal,
    MiCo,
    Twitter,
    CiteSeer,
    Wiki_Vote,
    YouTube,
    Friendster,
    Invalid
};

constexpr long long Patents_tri_cnt = 7515023LL;
constexpr long long LiveJournal_tri_cnt = 177820130LL;
constexpr long long MiCo_tri_cnt = 12534960LL;
constexpr long long CiteSeer_tri_cnt = 1166LL;
constexpr long long Wiki_Vote_tri_cnt = 608389LL;
constexpr long long Orkut_tri_cnt = 627584181LL;
constexpr long long Twitter_tri_cnt = 34824916864LL;
constexpr long long YouTube_tri_cnt = 103017122LL;
constexpr long long Friendster_tri_cnt = 4173724142LL;

class DataLoader {
public:

    bool load_data(Graph* &g, DataType type, const char* path, bool binary_input = false, int oriented_type = 0);
        // binary_input: binary graph input if true, otherwise text
        // pattern_diameter means max distance between two vertex in graph
        // oriented_type is used to reorder dataset
        // oriented_type == 0 do nothing
        //               == 1 high degree first
        //               == 2 low degree first

    bool load_labeled_data(LabeledGraph* &g, DataType type, const char* path);
        
    bool load_data(Graph* &g, int clique_size);

    bool fast_load(Graph* &g, const char* path);

    bool load_complete(Graph* &g, int clique_size);

private:
    static bool cmp_pair(std::pair<int,int>a, std::pair<int,int>b);
    static bool cmp_tuple(std::tuple<int,int,int>a, std::tuple<int,int,int>b);
    static bool cmp_label(std::pair<int,int> a,std::pair<int,int> b);
    static bool cmp_degree_gt(std::pair<int,int> a,std::pair<int,int> b);
    static bool cmp_degree_lt(std::pair<int,int> a,std::pair<int,int> b);

    long long comb(int n,int k);
    bool general_load_data(Graph* &g, DataType type, const char* path, bool binary_input, int oriented_type = 0);
    bool twitter_load_data(Graph* &g, DataType type, const char* path, int oriented_type = 0);
    bool general_load_labeled_data(LabeledGraph* &g, DataType type, const char* path);

    std::unordered_map<uint32_t, uint32_t> id;
    std::unordered_map<uint32_t, uint32_t> label;
};
