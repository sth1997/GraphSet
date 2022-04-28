#pragma once
#include "schedule.h"
#include "vertex_set.h"
#include <assert.h>
#include <omp.h>
#include <set>
#include <algorithm>


class Graphmpi;
class Graph {
public:
    int v_cnt; // number of vertex
    long long e_cnt; // number of edge
    long long tri_cnt; // number of triangle
    double max_running_time = 60 * 60 * 24; // second
    bool is_local_graph;

    int *edge; // edges
    long long *vertex; // v_i's neighbor is in edge[ vertex[i], vertex[i+1]-1]
    
    Graph() {
        v_cnt = 0;
        e_cnt = 0;
        edge = nullptr;
        vertex = nullptr;
        is_local_graph = false;
    }

    virtual ~Graph() { 
        if(edge != nullptr) delete[] edge;
        if(vertex != nullptr) delete[] vertex;
    }

    int intersection_size(int v1,int v2);
    int intersection_size_mpi(int v1,int v2);
    int intersection_size_clique(int v1,int v2);

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
    long long pattern_matching(const Schedule& schedule, int thread_count, bool clique = false);

    //general pattern matching algorithm with multi thread ans multi process
    long long pattern_matching_mpi(const Schedule& schedule, int thread_count, bool clique = false);
private:
    friend Graphmpi;
    void tc_mt(long long * global_ans);

    // void use_local_graph(const Schedule& schedule, VertexSet* vertex_set, VertexSet& subtraction_set, VertexSet& tmp_set, long long &local_ans, int depth, int u, int v, int *loop_data_ptr, int loop_size);

    void get_edge_index(int v, long long& l, long long& r) const;

    void pattern_matching_func(const Schedule& schedule, VertexSet* vertex_set, VertexSet& subtraction_set, long long& local_ans, int depth, bool clique = false);

    void pattern_matching_aggressive_func(const Schedule& schedule, VertexSet* vertex_set, VertexSet& subtraction_set, VertexSet& tmp_set, long long& local_ans, int depth);

    void pattern_matching_aggressive_func_mpi(const Schedule& schedule, VertexSet* vertex_set, VertexSet& subtraction_set, VertexSet &tmp_set, long long& local_ans, int depth);

    void clique_matching_func(const Schedule& schedule, VertexSet* vertex_set, Bitmap* bs, long long& local_ans, int depth);

};

// in reduce_edge.cpp
void reduce_edges_for_clique(Graph &g);

// struct Local_Graph : public Graph {
// private:
//     void get_edge_index(int v, unsigned int &l, unsigned int &r) const override{
//         l = vertex[v];
//         // r = vertex[v + 1];
//         r = vertex[v] + deg[v];
//     }

// public:

//     typedef unsigned long long adj_t;
//     int *deg;
//     int *_deg;
//     adj_t *adjmat; 

//     inline bool is_adj(int i, int j) {
//         return bool((adjmat[((i-1) * v_cnt + j) / (sizeof(adj_t) * 8)] >> (j % (sizeof(adj_t) * 8))) & 1);
//     }
    
//     inline void set_adj(int i, int j, int v){
//         adjmat[((i-1) * v_cnt + j) / (sizeof(adj_t) * 8) ] |= (v << (j % (sizeof(adj_t) * 8)));
//     }

//     Local_Graph() = delete;

 
//     Local_Graph(const Graph &g, int _u, int _v, int* intersection_set, int intersection_size, int &new_loop) : Graph() {

//         // adjmat = new adj_t[v_cnt * v_cnt / (sizeof(adj_t) * 8)];
//         // std::fill(adjmat, adjmat + v_cnt * v_cnt / (sizeof(adj_t) * 8), 0);

//         // u & v (u > v)
        
//         v_cnt = std::lower_bound(intersection_set, intersection_set + intersection_size, _v) - intersection_set;
        
//         new_loop = v_cnt;

//         e_cnt = 0;

//         vertex = new unsigned int[(v_cnt + 1)];
//         deg = new int[(v_cnt + 1)];
//         // printf("deg create:%p\n", _deg = deg);
//         edge = new int[v_cnt * v_cnt + 1];

//         // #pragma unroll
//         // for(int i = a; i < intersection_size; i++){
//         //     tmp[i + 2] = intersection_set[i];
//         // }
//         // tmp[a + 1] = _u;
//         // #pragma unroll
//         // for(int i = b; i < a; i++){
//         //     tmp[i + 1] = intersection_set[i];
//         // }
//         // tmp[b] = _v;
//         // #pragma unroll
//         // for(int i = 0; i < b; i++){
//         //     tmp[i] = intersection_set[i];
//         // }
//         // #pragma unroll
//         // for(int i = 1; i < v_cnt; i++){
//         //     assert(tmp[i - 1] < tmp[i]);
//         // }

//         int *tmp = intersection_set;

//         for(int v_i = 0; v_i < v_cnt; v_i++) {
//             int v = tmp[v_i];
//             vertex[v_i] = e_cnt;
//             // for(int v_j = 0; v_j < v_i; v_j++){
//             //     if(*std::lower_bound(&g.edge[g.vertex[v]], &g.edge[g.vertex[v + 1]], tmp[v_i]) == tmp[v_i]){
//             //         edge[e_cnt++] = v_j;
//             //     }
//             // }
//             int v_j = 0;
//             for(int e = g.vertex[v]; e < g.vertex[v + 1]; e++){
//                 int j = g.edge[e];
//                 while(v_j < v_i && tmp[v_j] < j) v_j++;
//                 if(v_j >= v_i) break;
//                 if(tmp[v_j] == j) {
//                     // set_adj(v_i , v_j, 1);
//                     edge[e_cnt++] = v_j ; // only only side is enough (v_i --> v_j , v_i > v_j)
//                 }
//             }
//             deg[v_i] = e_cnt - vertex[v_i];
//         }
//         vertex[v_cnt] = e_cnt;
//         is_local_graph = true;

//     }
//     ~Local_Graph(){
//         // printf("deg %d delete:%p deg create:%p\n", omp_get_thread_num() ,deg, _deg);
//         fflush(stdout);
//         delete[] deg;
//         // delete[] adjmat;
//     }


    
// };