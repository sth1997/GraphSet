#include "../include/graph.h"
#include "../include/graphmpi.h"
#include "../include/vertex_set.h"
#include "../include/common.h"
#include "../include/motif_generator.h"
#include "timeinterval.h"
#include <cstdio>
#include <chrono>
#include <sys/time.h>
#include <unistd.h>
#include <cstdlib>
#include <omp.h>
#include <algorithm>
#include <cstring>
#include <mpi.h>
#include <atomic>
#include <queue>
#include <iostream>

int Graph::intersection_size(v_index_t v1, v_index_t v2) {
    e_index_t l1, r1;
    get_edge_index(v1, l1, r1);
    e_index_t l2, r2;
    get_edge_index(v2, l2, r2);
    int ans = 0;
    while(l1 < r1 && l2 < r2) {
        if(edge[l1] < edge[l2]) {
            ++l1;
        }
        else {
            if(edge[l2] < edge[l1]) {
                ++l2;
            }
            else {
                ++l1;
                ++l2;
                ++ans;
            }
        }
    }
    return ans;
}

int Graph::intersection_size_mpi(v_index_t v1, v_index_t v2) {
    Graphmpi &gm = Graphmpi::getinstance();
    int ans = 0;
    if (gm.include(v2))
        return intersection_size(v1, v2);
    e_index_t l1, r1;
    get_edge_index(v1, l1, r1);
    int *data = gm.getneighbor(v2);
    for (e_index_t l2 = 0; l1 < r1 && ~data[l2];) {
        if(edge[l1] < data[l2]) {
            ++l1;
        }
        else if(edge[l1] > data[l2]) {
            ++l2;
        }
        else {
            ++l1;
            ++l2;
            ++ans;
        }
    }
    return ans;
}

int Graph::intersection_size_clique(v_index_t v1,v_index_t v2) {
    e_index_t l1, r1;
    get_edge_index(v1, l1, r1);
    e_index_t l2, r2;
    get_edge_index(v2, l2, r2);
    v_index_t min_vertex = v2;
    int ans = 0;
    if (edge[l1] >= min_vertex || edge[l2] >= min_vertex)
        return 0;
    while(l1 < r1 && l2 < r2) {
        if(edge[l1] < edge[l2]) {
            if (edge[++l1] >= min_vertex)
                break;
        }
        else {
            if(edge[l2] < edge[l1]) {
                if (edge[++l2] >= min_vertex)
                    break;
            }
            else {
                ++ans;
                if (edge[++l1] >= min_vertex)
                    break;
                if (edge[++l2] >= min_vertex)
                    break;
            }
        }
    }
    return ans;
}

long long Graph::triangle_counting() {
    long long ans = 0;
    for(int v = 0; v < v_cnt; ++v) {
        // for v in G
        e_index_t l, r;
        get_edge_index(v, l, r);
        for(e_index_t v1 = l; v1 < r; ++v1) {
            //for v1 in N(v)
            ans += intersection_size(v,edge[v1]);
        }
    }
    ans /= 6;
    return ans;
}

long long Graph::triangle_counting_mt(int thread_count) {
    long long ans = 0;
#pragma omp parallel num_threads(thread_count)
    {
        tc_mt(&ans);
    }
    return ans;
}

void Graph::tc_mt(long long *global_ans) {
    long long my_ans = 0;
    #pragma omp for schedule(dynamic)
    for(int v = 0; v < v_cnt; ++v) {
        // for v in G
        e_index_t l, r;
        get_edge_index(v, l, r);
        for(e_index_t v1 = l; v1 < r; ++v1) {
            if (v <= edge[v1])
                break;
            //for v1 in N(v)
            my_ans += intersection_size_clique(v,edge[v1]);
        }
    }
    #pragma omp critical
    {
        *global_ans += my_ans;
    }
}

long long Graph::triangle_counting_mpi(int thread_count) {
    /*int mynodel, mynoder;
    long long tot_ans;
    Graphmpi &gm = Graphmpi::getinstance();
#pragma omp parallel num_threads(thread_count)
    {
#pragma omp master
        {
            auto k = gm.init(thread_count, this);
            mynodel = k.first;
            mynoder = k.second;
        }
#pragma omp barrier //mynodel have to be calculated before running other threads
#pragma omp master
        {
            tot_ans = gm.runmajor();
        }
        long long thread_ans = 0;
#pragma omp for schedule(dynamic) nowait
        for(int v = mynodel; v < mynoder; v++) {
            // for v in G
            int l, r;
            get_edge_index(v, l, r);
            for(int v1 = l; v1 < r; ++v1) {
                //for v1 in N(v)
                thread_ans += intersection_size_mpi(v, edge[v1]);
            }
        }
        //gm.idle(thread_ans); 
    }
    return tot_ans / 6ll;*/
    return -1;
}

void Graph::get_edge_index(v_index_t v, e_index_t & l, e_index_t& r) const
{
    l = vertex[v];
    r = vertex[v + 1];
}

template <typename T>
bool binary_search(const T data[], int n, const T& target) {
    int mid, l = 0, r = n - 1;
    while (l <= r) {
        mid = (l + r) >> 1;
        if (data[mid] < target) {
            l = mid + 1;
        } else if (data[mid] > target) {
            r = mid - 1;
        } else {
            return true;
        }
    }
    return false;
}

void Graph::remove_anti_edge_vertices(VertexSet& out_buf, const VertexSet& in_buf,
    const Schedule_IEP& sched, const VertexSet& partial_embedding, int vp) {

    out_buf.init();
    assert(&out_buf != &in_buf);
    auto d_out = out_buf.get_data_ptr();
    assert(d_out != nullptr);
    auto d_in = in_buf.get_data_ptr();
    // printf("d_out:%x d_in:%x\n", d_out, d_in);
    assert(d_out != d_in);
    int n_in = in_buf.get_size();
    int out_size = 0;
    
    for (int i = 0; i < n_in; i++) {
        bool produce_output = true;
        for (int u = 0; u < vp; ++u) {
            if (sched.get_adj_mat_ptr()[u * sched.get_size() + vp] != 0)
                continue;
            
            auto v = partial_embedding.get_data(u);
            e_index_t l, r;
            get_edge_index(v, l, r);
            int m = r - l; // m = |N(v)|

            if (binary_search(&edge[l], m, d_in[i])) {
                produce_output = false;
                break;
            }
        }
        if(produce_output) {
            d_out[out_size++] = d_in[i];
        }
    }
    out_buf.set_size(out_size);
}

void Graph::pattern_matching_func(const Schedule_IEP& schedule, VertexSet* vertex_set, VertexSet& subtraction_set, long long& local_ans, int depth, bool clique)
{
    int loop_set_prefix_id = schedule.get_loop_set_prefix_id(depth);
    auto &vset = vertex_set[loop_set_prefix_id];
    int loop_size = vset.get_size();
    if (loop_size <= 0)
        return;
    int* loop_data_ptr = vset.get_data_ptr();
    /*if (clique == true)
      {
      int last_vertex = subtraction_set.get_last();
    // The number of this vertex must be greater than the number of last vertex.
    loop_start = std::upper_bound(loop_data_ptr, loop_data_ptr + loop_size, last_vertex) - loop_data_ptr;
    }*/
    if (schedule.is_vertex_induced) {
        VertexSet &diff_buf = vertex_set[schedule.get_total_prefix_num() + depth];
        diff_buf.init();
        remove_anti_edge_vertices(diff_buf, vertex_set[loop_set_prefix_id], schedule, subtraction_set, depth);
        loop_data_ptr = diff_buf.get_data_ptr();
        loop_size = diff_buf.get_size();
        vset = diff_buf;
    }
    if (depth == schedule.get_size() - 1)
    {
        // TODO : try more kinds of calculation.
        // For example, we can maintain an ordered set, but it will cost more to maintain itself when entering or exiting recursion.
        if (clique == true)
            local_ans += loop_size;
        else if (loop_size > 0)
            local_ans += VertexSet::unordered_subtraction_size(vset, subtraction_set);
        return;
    }

    int last_vertex = subtraction_set.get_last();
    for (int i = 0; i < loop_size; ++i)
    {
        if (last_vertex <= loop_data_ptr[i] && clique == true)
            break;
        int vertex = loop_data_ptr[i];
        if (!clique)
            if (subtraction_set.has_data(vertex))
                continue;
        e_index_t l, r;
        get_edge_index(vertex, l, r);
        bool is_zero = false;
        for (int prefix_id = schedule.get_last(depth); prefix_id != -1; prefix_id = schedule.get_next(prefix_id))
        {
            vertex_set[prefix_id].build_vertex_set(schedule, vertex_set, &edge[l], (int)(r - l), prefix_id, vertex, clique);
            if( vertex_set[prefix_id].get_size() == 0) {
                is_zero = true;
                break;
            }
        }
        if( is_zero ) continue;
        //subtraction_set.insert_ans_sort(vertex);
        subtraction_set.push_back(vertex);
        pattern_matching_func(schedule, vertex_set, subtraction_set, local_ans, depth + 1, clique);
        subtraction_set.pop_back();
    }
}

long long Graph::pattern_matching(const Schedule_IEP& schedule, int thread_count, bool clique)
{
//    intersection_times_low = intersection_times_high = 0;
//    dep1_cnt = dep2_cnt = dep3_cnt = 0;
    long long global_ans = 0;
#pragma omp parallel num_threads(thread_count) reduction(+: global_ans)
    {
     //   double start_time = get_wall_time();
     //   double current_time;
        int* ans_buffer = new int[schedule.in_exclusion_optimize_vertex_id.size()];
        VertexSet* vertex_set = new VertexSet[schedule.get_total_prefix_num() + 10];
        VertexSet subtraction_set;
        VertexSet tmp_set;
        subtraction_set.init();
        long long local_ans = 0;
        // TODO : try different chunksize
#pragma omp for schedule(dynamic) nowait
        for (int vertex = 0; vertex < v_cnt; ++vertex)
        {
            e_index_t l, r;
            get_edge_index(vertex, l, r);
            for (int prefix_id = schedule.get_last(0); prefix_id != -1; prefix_id = schedule.get_next(prefix_id))
            {
                vertex_set[prefix_id].build_vertex_set(schedule, vertex_set, &edge[l], (int)(r - l), prefix_id);
            }
            //subtraction_set.insert_ans_sort(vertex);
            subtraction_set.push_back(vertex);
            //if (schedule.get_total_restrict_num() > 0 && clique == false)
            if(true) {
                pattern_matching_aggressive_func(schedule, vertex_set, subtraction_set, tmp_set,local_ans, 1, ans_buffer);
            }
            else
                pattern_matching_func(schedule, vertex_set, subtraction_set, local_ans, 1, clique);
            subtraction_set.pop_back();
            //printf("for %d %d\n", omp_get_thread_num(), vertex);
        }
        //double end_time = get_wall_time();
        //printf("my thread time %d %.6lf\n", omp_get_thread_num(), end_time - start_time);
        delete[] vertex_set;
        // TODO : Computing multiplicty for a pattern
        global_ans += local_ans;
        //printf("local_ans %d %lld\n", omp_get_thread_num(), local_ans);
        
    }
    return global_ans / schedule.get_in_exclusion_optimize_redundancy();
}

void Graph::clique_matching_func(const Schedule_IEP& schedule, VertexSet* vertex_set, Bitmap* bs, long long& local_ans, int depth) {
    int loop_set_prefix_id = schedule.get_loop_set_prefix_id(depth);
    int loop_size = vertex_set[loop_set_prefix_id].get_size();
    int* loop_data_ptr = vertex_set[loop_set_prefix_id].get_data_ptr();

    for (int i = 0; i < loop_size; ++i) {
        int vertex = loop_data_ptr[i];
        e_index_t l, r;
        get_edge_index(vertex, l, r);
        int prefix_id = schedule.get_last(depth);// only one prefix
        if(depth == schedule.get_size() - 2)
            vertex_set[prefix_id].build_vertex_set_bs_only_size(schedule, vertex_set, bs, &edge[l], (int)(r - l), prefix_id, depth);
        else 
            vertex_set[prefix_id].build_vertex_set_bs(schedule, vertex_set, bs, &edge[l], (int)(r - l), prefix_id, depth);

        if(depth == schedule.get_size() - 2) {
            local_ans += vertex_set[prefix_id].get_size();
            continue;
        }
        if(vertex_set[prefix_id].get_size() == 0)
            continue;
        clique_matching_func(schedule, vertex_set, bs, local_ans, depth + 1);
        int *_data = vertex_set[prefix_id].get_data_ptr(), _size = vertex_set[prefix_id].get_size();
        for(int j = 0; j < _size; j++) bs->dec(_data[j]);
    }
}

void Graph::pattern_matching_aggressive_func(const Schedule_IEP& schedule, VertexSet* vertex_set, VertexSet& subtraction_set, VertexSet& tmp_set, long long& local_ans, int depth, int* ans_buffer)
{
    int loop_set_prefix_id = schedule.get_loop_set_prefix_id(depth);
    auto vset = &vertex_set[loop_set_prefix_id];

    int loop_size = (*vset).get_size();
    if (loop_size <= 0)
        return;

    int* loop_data_ptr = (*vset).get_data_ptr();

    if (schedule.is_vertex_induced) {
        VertexSet &diff_buf = vertex_set[schedule.get_total_prefix_num() + depth];
        // printf("depth:%d loop_set_prefix_id = %d diff_buf: %d\n",depth, loop_set_prefix_id, schedule.get_total_prefix_num() + depth);
        remove_anti_edge_vertices(diff_buf, vertex_set[loop_set_prefix_id], schedule, subtraction_set, depth);
        loop_data_ptr = diff_buf.get_data_ptr();
        loop_size = diff_buf.get_size();
        vset = &diff_buf;
    }
    //Case: in_exclusion_optimize_num > 1
    if( depth == schedule.get_size() - schedule.get_in_exclusion_optimize_num() ) {
        
        int last_pos = -1;
        long long val;

        for(int i = 0; i < schedule.in_exclusion_optimize_vertex_id.size(); ++i) {
            if(schedule.in_exclusion_optimize_vertex_flag[i]) {
                ans_buffer[i] = vertex_set[schedule.in_exclusion_optimize_vertex_id[i]].get_size() - schedule.in_exclusion_optimize_vertex_coef[i];
            }
            else {
                ans_buffer[i] = VertexSet::unordered_subtraction_size(vertex_set[schedule.in_exclusion_optimize_vertex_id[i]], subtraction_set);
            }
        }

        for(int pos = 0; pos < schedule.in_exclusion_optimize_coef.size(); ++pos) {
            if(pos == last_pos + 1)
                val = ans_buffer[schedule.in_exclusion_optimize_ans_pos[pos]];
            else {
                if( val != 0)
                    val = val * ans_buffer[schedule.in_exclusion_optimize_ans_pos[pos]];
            }
            if(schedule.in_exclusion_optimize_flag[pos]) {
                last_pos = pos;
                local_ans += val * schedule.in_exclusion_optimize_coef[pos];
            }
        }

        return;
            
    }
    //Case: in_exclusion_optimize_num <= 1
    if (depth == schedule.get_size() - 1)
    {
        // TODO : try more kinds of calculation.
        // For example, we can maintain an ordered set, but it will cost more to maintain itself when entering or exiting recursion.
        if (schedule.get_total_restrict_num() > 0)
        {
            int min_vertex = v_cnt;
            for (int i = schedule.get_restrict_last(depth); i != -1; i = schedule.get_restrict_next(i))
                if (min_vertex > subtraction_set.get_data(schedule.get_restrict_index(i)))
                    min_vertex = subtraction_set.get_data(schedule.get_restrict_index(i));
            int size_after_restrict = std::lower_bound((*vset).get_data_ptr(), (*vset).get_data_ptr() + (*vset).get_size(), min_vertex) - (*vset).get_data_ptr();
            if (size_after_restrict > 0)
                local_ans += VertexSet::unordered_subtraction_size((*vset), subtraction_set, size_after_restrict);
        }
        else
            local_ans += VertexSet::unordered_subtraction_size((*vset), subtraction_set);
        return;
    }

/*    #pragma omp critical
    {
        if( depth == 1) ++dep1_cnt;
        if( depth == 2) ++dep2_cnt;
        if( depth == 3) ++dep3_cnt;
    }*/
    // TODO : min_vertex is also a loop invariant
    int min_vertex = v_cnt;
    for (int i = schedule.get_restrict_last(depth); i != -1; i = schedule.get_restrict_next(i))
        if (min_vertex > subtraction_set.get_data(schedule.get_restrict_index(i)))
            min_vertex = subtraction_set.get_data(schedule.get_restrict_index(i));
    for (int i = 0; i < loop_size; ++i)
    {
        if (min_vertex <= loop_data_ptr[i])
            break;
        int vertex = loop_data_ptr[i];
        if (subtraction_set.has_data(vertex))
            continue;
        e_index_t l, r;
        get_edge_index(vertex, l, r);
        bool is_zero = false;
        for (int prefix_id = schedule.get_last(depth); prefix_id != -1; prefix_id = schedule.get_next(prefix_id))
        {
            vertex_set[prefix_id].build_vertex_set(schedule, vertex_set, &edge[l], (int)(r - l), prefix_id, vertex);
            //if( vertex_set[prefix_id].get_size() == 0 && prefix_id < schedule.get_basic_prefix_num()) {
            if( vertex_set[prefix_id].get_size() == schedule.break_size[prefix_id]) {
                is_zero = true;
                break;
            }
        }
        if( is_zero ) continue;
        //subtraction_set.insert_ans_sort(vertex);
        subtraction_set.push_back(vertex);
        pattern_matching_aggressive_func(schedule, vertex_set, subtraction_set, tmp_set, local_ans, depth + 1, ans_buffer);
        subtraction_set.pop_back();
    }
}

long long Graph::pattern_matching_mpi(const Schedule_IEP& schedule, int thread_count, bool clique)
{
    Graphmpi &gm = Graphmpi::getinstance();
    long long global_ans = 0;
#pragma omp parallel num_threads(thread_count)
    {
#pragma omp master
        {
            gm.init(thread_count, this);
        }
#pragma omp barrier //mynodel have to be calculated before running other threads
#pragma omp master
        {
            global_ans = gm.runmajor();
        }
        if (omp_get_thread_num()) {
            int* ans_buffer = new int[schedule.in_exclusion_optimize_vertex_id.size()];
            VertexSet* vertex_set = new VertexSet[schedule.get_total_prefix_num()];
            long long local_ans = 0;
            VertexSet subtraction_set;
            VertexSet tmp_set;
            subtraction_set.init();
            auto match_start_vertex = [&](int vertex, int *data, int size) {
                for (int prefix_id = schedule.get_last(0); prefix_id != -1; prefix_id = schedule.get_next(prefix_id))
                {
                    vertex_set[prefix_id].build_vertex_set(schedule, vertex_set, data, size, prefix_id);
                }
                //subtraction_set.insert_ans_sort(vertex);
                subtraction_set.push_back(vertex);
                pattern_matching_aggressive_func(schedule, vertex_set, subtraction_set, tmp_set, local_ans, 1, ans_buffer);
                subtraction_set.pop_back();
            };
            for (std::pair<int, int> range;;)
            {
                if ((range = gm.get_vertex_range()).first == -1) break;
                //for (int vertex = v_cnt - range.second; vertex < v_cnt - range.first; vertex++) {//backwards slower than forwards
                for (int vertex = range.first; vertex < range.second; vertex++) {
                    e_index_t l, r;
                    get_edge_index(vertex, l, r);
                    match_start_vertex(vertex, edge + l, r - l);
                }
            }
            delete[] vertex_set;
            gm.report(local_ans);
            gm.end();
        }
    }
    return global_ans;
}

void Graph::pattern_matching_aggressive_func_mpi(const Schedule_IEP& schedule, VertexSet* vertex_set, VertexSet& subtraction_set, VertexSet &tmp_set, long long& local_ans, int depth)
{
    int loop_set_prefix_id = schedule.get_loop_set_prefix_id(depth);
    int loop_size = vertex_set[loop_set_prefix_id].get_size();
    if (loop_size <= 0)
        return;
    int* loop_data_ptr = vertex_set[loop_set_prefix_id].get_data_ptr();
    if (depth == schedule.get_size() - 1)
    {
        // TODO : try more kinds of calculation.
        // For example, we can maintain an ordered set, but it will cost more to maintain itself when entering or exiting recursion.
        if (schedule.get_total_restrict_num() > 0)
        {
            int min_vertex = v_cnt;
            for (int i = schedule.get_restrict_last(depth); i != -1; i = schedule.get_restrict_next(i))
                if (min_vertex > subtraction_set.get_data(schedule.get_restrict_index(i)))
                    min_vertex = subtraction_set.get_data(schedule.get_restrict_index(i));
            const VertexSet& vset = vertex_set[loop_set_prefix_id];
            int size_after_restrict = std::lower_bound(vset.get_data_ptr(), vset.get_data_ptr() + vset.get_size(), min_vertex) - vset.get_data_ptr();
            if (size_after_restrict > 0)
                local_ans += VertexSet::unordered_subtraction_size(vertex_set[loop_set_prefix_id], subtraction_set, size_after_restrict);
        }
        else
            local_ans += VertexSet::unordered_subtraction_size(vertex_set[loop_set_prefix_id], subtraction_set);
        return;
    }
    
    // TODO : min_vertex is also a loop invariant
    int min_vertex = v_cnt;
    for (int i = schedule.get_restrict_last(depth); i != -1; i = schedule.get_restrict_next(i))
        if (min_vertex > subtraction_set.get_data(schedule.get_restrict_index(i)))
            min_vertex = subtraction_set.get_data(schedule.get_restrict_index(i));
    for (int i = 0; i < loop_size; ++i)
    {
        if (min_vertex <= loop_data_ptr[i])
            break;
        int vertex = loop_data_ptr[i];
        if (subtraction_set.has_data(vertex))
            continue;
        int *data, size;
        Graphmpi &gm = Graphmpi::getinstance();
        //if (gm.include(vertex)) {
        if (true) {
            e_index_t l, r;
            get_edge_index(vertex, l, r);
            data = edge + l;
            size = r - l;
        }
        else {
            data = gm.getneighbor(vertex);
            size = gm.getdegree();
        }
        bool is_zero = false;
        for (int prefix_id = schedule.get_last(depth); prefix_id != -1; prefix_id = schedule.get_next(prefix_id))
        {
            vertex_set[prefix_id].build_vertex_set(schedule, vertex_set, data, size, prefix_id, vertex);
            if( vertex_set[prefix_id].get_size() == 0) {
                is_zero = true;
                break;
            }
        }
        if( is_zero ) continue;
        //subtraction_set.insert_ans_sort(vertex);
        subtraction_set.push_back(vertex);
        int* ans_buffer = new int[schedule.in_exclusion_optimize_vertex_id.size()];
        pattern_matching_aggressive_func(schedule, vertex_set, subtraction_set, tmp_set, local_ans, depth + 1, ans_buffer);
        subtraction_set.pop_back();
    }
}

void erase_edge(Graph &g) {
    int newe = 0;
    printf("start erasing edge\n");
    fflush(stdout);
    for(int v = 0; v < g.v_cnt; v++) {
        int l = newe;
        for(long long e = g.vertex[v]; e < g.vertex[v+1] && e < g.e_cnt; e++){
            if(g.edge[e] >= v) continue;
            g.edge[newe] = g.edge[e];
            newe++; 
        }
        g.vertex[v] = l;
    }
    g.vertex[g.v_cnt] = newe;
    g.e_cnt = newe;
    printf("newe: %ld\n", g.e_cnt);
}

// reduce two directed edge to one 
void reduce_edges_for_clique(Graph &g) {
    printf("Trying to reduce edge. the pattern is a clique.\n");
    erase_edge(g);
    printf("Finish reduce.\n");
}


void degree_orientation_init(Graph* original_g, Graph*& g) {
  g = new Graph();

  int* degree;
  int n = original_g->v_cnt;
  long long m = original_g->e_cnt;
  g->v_cnt = n;
  g->e_cnt = m >> 1;
  g->edge = new int [g->e_cnt];
  g->vertex = new int64_t[g->v_cnt + 1];

  degree = new int [n];
  for (int u = 0; u < n; ++u) {
    degree[u] = original_g->vertex[u + 1] - original_g->vertex[u];
  }

  int64_t ptr = 0;
  int max_degree = 0;
  for (int u = 0; u < n; ++u) {
    g->vertex[u] = ptr;
    for (int64_t i = original_g->vertex[u]; i < original_g->vertex[u + 1];
         ++i) {
      int v = original_g->edge[i];
      if ((degree[u] < degree[v]) || (degree[u] == degree[v] && u < v)) {
        g->edge[ptr++] = v;
      }
    }
    if (max_degree < ptr - g->vertex[u]) {
      max_degree = ptr - g->vertex[u];
    }
  }
  g->vertex[n] = ptr;
  assert(ptr == g->e_cnt);
  // partition_num = (max_degree - 1) / 32 + 1;
  // assert(max_degree < THREADS_PER_BLOCK);
  printf("Maximum degree after orientation: %d\n", max_degree);

  delete[] degree;
}


void degeneracy_orientation_init(Graph* original_g, Graph*& g) {
  g = new Graph();

  // 这里用 vector<int> node_with_degree[x] 来装当前度数为 x
  // 的所有节点，做到了线性求 k-core 序
  int* degree;
  int* order;
  int* vector_ptr;
  bool* in_vector;
  int n = original_g->v_cnt;
  long long m = original_g->e_cnt;
  std::vector<std::vector<int>> node_with_degree(n, std::vector<int>());
  g->v_cnt = n;
  g->e_cnt = m >> 1;
  g->edge = new int [g->e_cnt] ;
  g->vertex = new int64_t [g->v_cnt + 1];

  degree = new int [n];
  order = new int [n];
  in_vector = new bool [n];
  vector_ptr = new int [n];

  for (int u = 0; u < n; ++u) {
    in_vector[u] = true;
    degree[u] = original_g->vertex[u + 1] - original_g->vertex[u];
    node_with_degree[degree[u]].push_back(u);
  }

  for (int i = 0; i < n; ++i) {
    vector_ptr[i] = 0;
  }

  // 精巧的实现
  int order_ptr = 0;
  for (int current_min_degree = 0; current_min_degree < n;
       ++current_min_degree) {
    bool back = false;
    for (int& i = vector_ptr[current_min_degree];
         i < node_with_degree[current_min_degree].size(); ++i) {
      int u = node_with_degree[current_min_degree][i];
      if (in_vector[u]) {
        order[u] = order_ptr++;
        in_vector[u] = false;
        for (int64_t j = original_g->vertex[u]; j < original_g->vertex[u + 1];
             ++j) {
          int v = original_g->edge[j];
          if (in_vector[v]) {
            node_with_degree[--degree[v]].push_back(v);
            if (degree[v] == current_min_degree - 1) {
              back = true;
            }
          }
        }
      }
    }
    if (back) {
      // 由于 for 循环里还要 +1，这里实际上只减了 1
      current_min_degree -= 2;
    }
  }

  int64_t ptr = 0;
  int max_degree = 0;
  for (int u = 0; u < n; ++u) {
    g->vertex[u] = ptr;
    for (int64_t i = original_g->vertex[u]; i < original_g->vertex[u + 1];
         ++i) {
      int v = original_g->edge[i];
      if (order[u] < order[v]) {
        g->edge[ptr++] = v;
      }
    }
    if (max_degree < ptr - g->vertex[u]) {
      max_degree = ptr - g->vertex[u];
    }
  }
  g->vertex[n] = ptr;
  assert(ptr == g->e_cnt);
  // partition_num = (max_degree - 1) / 32 + 1;
  printf("Maximum vertex degree after orientation: %d\n", max_degree);

  delete[] degree;
  delete[] order;
  delete[] in_vector;
  delete[] vector_ptr;
}


void Graph::motif_counting(int pattern_size, int thread_count) {

    TimeInterval allTime;

    allTime.check();

    MotifGenerator mg(pattern_size);
    std::vector<Pattern> motifs = mg.generate();
    for (int i = 0; i < motifs.size(); ++i) {
        Pattern p = motifs[i];

        printf("pattern = \n");
        p.print();
    
        printf("max intersection size %d\n", VertexSet::max_intersection_size);
        bool is_pattern_valid;
        bool use_in_exclusion_optimize = true;

        Schedule_IEP schedule_iep(p, is_pattern_valid, 1, 1, use_in_exclusion_optimize, v_cnt, e_cnt, tri_cnt, true);

        if (!is_pattern_valid) {
            printf("pattern is invalid!\n");
            continue;
        }

        long long ans = this->pattern_matching(schedule_iep, thread_count);

        printf("ans: %lld\n",ans);

        allTime.print("Total time cost");
    }
}

int32_t get_intersection_size(const v_index_t *a, int32_t na, const v_index_t *b, int32_t nb) {
    int32_t ans = 0;
    int32_t i = 0, j = 0;
    while(i < na) {
        if(i >= 1) {
            assert(a[i] > a[i-1]);
        }
        while(j < nb && b[j] < a[i]) j++;
        if(j == nb) break;
        if(a[i] == b[j]) ans++;
        i++;
    }
    return ans;
}


void Graph::motif_counting_3(int thread_count) {
    int64_t tri_cnt = 0, wedge_cnt = 0;

    uint32_t *edge_from = new uint32_t[e_cnt];
    for (uint32_t i = 0; i < v_cnt; ++i)
        for (e_index_t j = vertex[i]; j < vertex[i+1]; ++j)
            edge_from[j] = i;
    
    auto t1 = std::chrono::system_clock::now();


    #pragma omp parallel for schedule(dynamic) num_threads(thread_count) reduction(+:tri_cnt) reduction(+:wedge_cnt)
    for(e_index_t i = 0; i < e_cnt; i++) {
        v_index_t v0 = edge_from[i], v1 = edge[i];
        e_index_t l0 = vertex[v0], r0 = vertex[v0+1];
         
        if(i == l0) {
            e_index_t d = r0 - l0;
            wedge_cnt += d * (d - 1) / 2;
        }
        
        if(v0 <= v1)
            continue;
    
        e_index_t l1 = vertex[v1],r1 = vertex[v1+1];
        tri_cnt += get_intersection_size(&edge[l0],r0-l0,&edge[l1],r1-l1);
    }

    wedge_cnt -= tri_cnt;

    tri_cnt /= 3;
    auto t2 = std::chrono::system_clock::now();
    double time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() * 1e-6;

    printf("triangle: %ld wedge: %ld time:%.6lf\n", tri_cnt, wedge_cnt, time);
}

