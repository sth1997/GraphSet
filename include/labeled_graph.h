#pragma once
#include "schedule.h"
#include "vertex_set.h"
#include <assert.h>
#include <set>
#include <vector>
#include <unordered_map>

class LabeledGraph {
public:
    int v_cnt; // number of vertex
    unsigned int e_cnt; // number of edge
    unsigned int l_cnt; // number of label
    long long tri_cnt; // number of triangle
    double max_running_time = 60 * 60 * 24; // second

    int *edge; // edges
    int *v_label;
    std::unordered_map<uint32_t, uint32_t> label_map;

    unsigned int *labeled_vertex; // v_i's neighbor whose label is c is in edge[ vertex[i * maxlabel + c], vertex[i * maxlabel + c + 1]-1]

    LabeledGraph() {
        v_cnt = 0;
        e_cnt = 0;
        edge = nullptr;
        v_label = nullptr;
        labeled_vertex = nullptr;
    }

    ~LabeledGraph() {
        if(edge != nullptr) delete[] edge;
        if(v_label != nullptr) delete[] v_label;
        if(labeled_vertex != nullptr) delete[] labeled_vertex;
    }
    
    void get_edge_index(int v, int label, unsigned int& l, unsigned int& r) const;
    long long get_support_pattern_matching(VertexSet* vertex_set, VertexSet& subtraction_set, const Schedule& schedule, const char* p_label, std::vector<std::set<int> >& fsm_set);
    int fsm(int max_edge, long long _min_support, int thread_count); // return the number of frequent labeled patterns with max_edge edges
private:
    //bool 返回值代表是否有至少匹配到一个子图，若匹配到，上一层就可以加入set
    bool get_support_pattern_matching_aggressive_func(const Schedule& schedule, const char* p_label, VertexSet* vertex_set, VertexSet& subtraction_set, std::vector<std::set<int> >& fsm_set, int depth);
    int fsm_cnt;
    unsigned char* is_frequent; //bit vector
    unsigned int* pattern_is_frequent_index; //每个unlabeled pattern对应的所有labeled pattern在is_frequent中的起始位置
    int min_support;
    void traverse_all_labeled_patterns(const Schedule* schedules, char* all_p_label, char* p_label, const int* mapping_start_idx, const int* mappings, int s_id, int depth, int mapping_start_idx_pos, size_t& all_p_label_idx);
};