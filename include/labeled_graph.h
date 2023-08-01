#pragma once
#include "schedule_IEP.h"
#include "vertex_set.h"
#include <assert.h>
#include <set>
#include <vector>
#include <cstdint>
#include <unordered_set>
#include <unordered_map>
#include <cstdint>



class LabeledGraph {
public:
    int v_cnt; // number of vertex
    unsigned int e_cnt; // number of edge
    unsigned int l_cnt; // number of label
    long long tri_cnt; // number of triangle
    double max_running_time = 60 * 60 * 24; // second

    int *edge; // edges
    int *v_label;
    int *label_frequency; //每个label的出现次数
    std::unordered_map<uint32_t, uint32_t> label_map;

    unsigned int *labeled_vertex; // v_i's neighbor whose label is c is in edge[ vertex[i * maxlabel + c], vertex[i * maxlabel + c + 1]-1]
    unsigned int *label_start_idx; //所有节点默认按照label排序，[label_start_idx[i], label_start_idx[i + 1]) 是label为i的所有节点

    LabeledGraph() {
        v_cnt = 0;
        e_cnt = 0;
        edge = nullptr;
        v_label = nullptr;
        labeled_vertex = nullptr;
        label_start_idx = nullptr;
    }

    ~LabeledGraph() {
        if(edge != nullptr) delete[] edge;
        if(v_label != nullptr) delete[] v_label;
        if(labeled_vertex != nullptr) delete[] labeled_vertex;
        if(label_start_idx != nullptr) delete[] label_start_idx;
    }
    
    void get_edge_index(int v, int label, unsigned int& l, unsigned int& r) const;
    long long get_support_pattern_matching(VertexSet* vertex_set, VertexSet& subtraction_set, const Schedule_IEP& schedule, const char* p_label, std::vector<std::set<int> >& fsm_set, long long min_support) const ;
    void get_support_pattern_matching_vertex(int vertex, VertexSet* vertex_set, VertexSet& subtraction_set, const Schedule_IEP& schedule, const char* p_label, std::vector<std::set<int> >& fsm_set, int min_support) const; 
    void get_fsm_necessary_info(std::vector<Pattern>& patterns, int max_edge, Schedule_IEP*& schedules, int& schedules_num, int*& mapping_start_idx, int*& mappings, unsigned int*& pattern_is_frequent_index, unsigned int*& is_frequent) const;
    void traverse_all_labeled_patterns(const Schedule_IEP* schedules, char* all_p_label, char* p_label, const int* mapping_start_idx, const int* mappings, const unsigned int* pattern_is_frequent_index, const unsigned int* is_frequent, int s_id, int depth, int mapping_start_idx_pos, size_t& all_p_label_idx) const;
    int fsm(int max_edge, long long _min_support, double *time_out = nullptr); // return the number of frequent labeled patterns with max_edge edges
    int fsm_pattern_matching(int job_start, int job_end, const Schedule_IEP &schedule, const char *all_p_label, std::vector<std::vector<int> > &automorphisms, unsigned int* is_frequent, unsigned int& pattern_is_frequent_index, int max_edge, int min_support) const;
    int fsm_vertex(int max_edge, long long _min_support, double *time_out = nullptr); 
    int fsm_pattern_matching_vertex(int job_id, const Schedule_IEP &schedule, const char *p_label, std::vector<std::vector<int> > &automorphisms, unsigned int* is_frequent, unsigned int& pattern_is_frequent_index,  int max_edge, int min_support) const;
private:
    //bool 返回值代表是否有至少匹配到一个子图，若匹配到，上一层就可以加入set
    bool get_support_pattern_matching_aggressive_func(const Schedule_IEP& schedule, const char* p_label, VertexSet* vertex_set, VertexSet& subtraction_set, std::vector<std::set<int> >& fsm_set, int depth) const;
    int fsm_cnt;
};