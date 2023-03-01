#pragma once
#include "schedule_IEP.h"
#include "vertex_set.h"
#include <cassert>
#include <cstdint>

typedef int32_t v_index_t;
typedef int64_t e_index_t;

constexpr int chunk_size = 2000;

class Graphmpi;
class Graph {
public:
    v_index_t v_cnt; // number of vertex
    e_index_t e_cnt; // number of edge
    long long tri_cnt; // number of triangle
    double max_running_time = 60 * 60 * 24; // second
    v_index_t *edge, *edge_from; // edges
    e_index_t *vertex; // v_i's neighbor is in edge[ vertex[i], vertex[i+1]-1]
    
    Graph() {
        v_cnt = 0;
        e_cnt = 0LL;
        edge = nullptr;
        vertex = nullptr;
        edge_from = nullptr;
    }

    ~Graph() {
        if(edge != nullptr) delete[] edge;
        if(vertex != nullptr) delete[] vertex;
        if (edge_from != nullptr) delete[] edge_from;
    }

    int intersection_size(v_index_t v1,v_index_t v2);
    int intersection_size_mpi(v_index_t v1,v_index_t v2);
    int intersection_size_clique(v_index_t v1,v_index_t v2);
    void build_reverse_edges();

/*    long long intersection_times_low;
    long long intersection_times_high;
    long long dep1_cnt;
    long long dep2_cnt;
    long long dep3_cnt;
*/
    //single thread triangle counting
    long long triangle_counting();
    
    //multi thread triangle counting
    long long triangle_counting_mt(int thread_count);

    //multi thread triangle counting with mpi
    long long triangle_counting_mpi(int thread_count);

    //general pattern matching algorithm with multi thread
    long long pattern_matching(const Schedule_IEP& schedule, int thread_count, bool clique = false);

    //general pattern matching algorithm with multi thread ans multi process
    long long pattern_matching_mpi(const Schedule_IEP& schedule, int thread_count, bool clique = false);

    // naive motif counting
    void motif_counting(int pattern_size, int thread_count);

    // hand optimized 3-motif counting
    void motif_counting_3(int thread_count);

    // internal use only
    long long pattern_matching_edge_task(const Schedule_IEP& schedule, int edge_id,
        VertexSet vertex_sets[], VertexSet& partial_embedding, VertexSet& tmp_set, int ans_buffer[]);
    
    void get_third_layer_size(const Schedule_IEP& schedule, int *count) const;

    void reorder_edge(const Schedule_IEP& schedule, e_index_t * new_order, e_index_t * task_start, int total_devices) const;

private:
    friend Graphmpi;
    void tc_mt(long long * global_ans);

    void remove_anti_edge_vertices(VertexSet& out_buf, const VertexSet& in_buf, const Schedule_IEP& sched, const VertexSet& partial_embedding, int vp);

    void get_edge_index(v_index_t v, e_index_t& l, e_index_t& r) const;

    void clique_matching_func(const Schedule_IEP& schedule, VertexSet* vertex_set, Bitmap* bs, long long& local_ans, int depth);

    void pattern_matching_func(const Schedule_IEP& schedule, VertexSet* vertex_set, VertexSet& subtraction_set, long long& local_ans, int depth, bool clique = false);

    void pattern_matching_aggressive_func(const Schedule_IEP& schedule, VertexSet* vertex_set, VertexSet& subtraction_set, VertexSet& tmp_set, long long& local_ans, int depth, int* ans_buffer);

    void pattern_matching_aggressive_func_mpi(const Schedule_IEP& schedule, VertexSet* vertex_set, VertexSet& subtraction_set, VertexSet &tmp_set, long long& local_ans, int depth);
    
};

void reduce_edges_for_clique(Graph &g);

void degree_orientation_init(Graph* original_g, Graph*& g);

void degeneracy_orientation_init(Graph* original_g, Graph*& g);
