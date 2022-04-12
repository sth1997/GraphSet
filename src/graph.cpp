#include "../include/graph.h"
#include "../include/graphmpi.h"
#include "../include/vertex_set.h"
#include "../include/common.h"
#include <cstdio>
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

void Graph::build_reverse_edges() {
    edge_from = new int[e_cnt];
    for (int u = 0; u < v_cnt; ++u)
        for (int v = vertex[u]; v < vertex[u + 1]; ++v)
            edge_from[v] = u;
}

int Graph::intersection_size(int v1,int v2) {
    unsigned int l1, r1;
    get_edge_index(v1, l1, r1);
    unsigned int l2, r2;
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

int Graph::intersection_size_mpi(int v1, int v2) {
    Graphmpi &gm = Graphmpi::getinstance();
    int ans = 0;
    if (gm.include(v2))
        return intersection_size(v1, v2);
    unsigned int l1, r1;
    get_edge_index(v1, l1, r1);
    int *data = gm.getneighbor(v2);
    for (int l2 = 0; l1 < r1 && ~data[l2];) {
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

int Graph::intersection_size_clique(int v1,int v2) {
    unsigned int l1, r1;
    get_edge_index(v1, l1, r1);
    unsigned int l2, r2;
    get_edge_index(v2, l2, r2);
    int min_vertex = v2;
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
        unsigned int l, r;
        get_edge_index(v, l, r);
        for(unsigned int v1 = l; v1 < r; ++v1) {
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
        unsigned int l, r;
        get_edge_index(v, l, r);
        for(unsigned int v1 = l; v1 < r; ++v1) {
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

void Graph::get_edge_index(int v, unsigned int& l, unsigned int& r) const
{
    l = vertex[v];
    r = vertex[v + 1];
}

void Graph::pattern_matching_func(const Schedule_IEP& schedule, VertexSet* vertex_set, VertexSet& subtraction_set, long long& local_ans, int depth, bool clique)
{
    int loop_set_prefix_id = schedule.get_loop_set_prefix_id(depth);
    int loop_size = vertex_set[loop_set_prefix_id].get_size();
    if (loop_size <= 0)
        return;
    int* loop_data_ptr = vertex_set[loop_set_prefix_id].get_data_ptr();
    /*if (clique == true)
      {
      int last_vertex = subtraction_set.get_last();
    // The number of this vertex must be greater than the number of last vertex.
    loop_start = std::upper_bound(loop_data_ptr, loop_data_ptr + loop_size, last_vertex) - loop_data_ptr;
    }*/
    if (depth == schedule.get_size() - 1)
    {
        // TODO : try more kinds of calculation.
        // For example, we can maintain an ordered set, but it will cost more to maintain itself when entering or exiting recursion.
        if (clique == true)
            local_ans += loop_size;
        else if (loop_size > 0)
            local_ans += VertexSet::unordered_subtraction_size(vertex_set[loop_set_prefix_id], subtraction_set);
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
        unsigned int l, r;
        get_edge_index(vertex, l, r);
        bool is_zero = false;
        for (int prefix_id = schedule.get_last(depth); prefix_id != -1; prefix_id = schedule.get_next(prefix_id))
        {
            vertex_set[prefix_id].build_vertex_set(schedule, vertex_set, &edge[l], (int)r - l, prefix_id, vertex, clique);
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
        VertexSet* vertex_set = new VertexSet[schedule.get_total_prefix_num()];
        VertexSet subtraction_set;
        VertexSet tmp_set;
        subtraction_set.init();
        long long local_ans = 0;
        // TODO : try different chunksize
#pragma omp for schedule(dynamic) nowait
        for (int vertex = 0; vertex < v_cnt; ++vertex)
        {
            unsigned int l, r;
            get_edge_index(vertex, l, r);
            for (int prefix_id = schedule.get_last(0); prefix_id != -1; prefix_id = schedule.get_next(prefix_id))
            {
                vertex_set[prefix_id].build_vertex_set(schedule, vertex_set, &edge[l], (int)r - l, prefix_id);
            }
            //subtraction_set.insert_ans_sort(vertex);
            subtraction_set.push_back(vertex);
            //if (schedule.get_total_restrict_num() > 0 && clique == false)
            if(true)
                pattern_matching_aggressive_func(schedule, vertex_set, subtraction_set, tmp_set, local_ans, 1, ans_buffer);
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

long long Graph::pattern_matching_edge_task(const Schedule_IEP &schedule, int edge_id, VertexSet *vertex_sets, VertexSet &partial_embedding, VertexSet &tmp_set, int *ans_buffer){
    int v0, v1, l, r;
    v0 = edge_from[edge_id];
    v1 = edge[edge_id];
    if (schedule.get_restrict_last(1) != -1 && v0 <= v1)
        return 0;
    
    l = vertex[v0], r = vertex[v0 + 1];
    for (int prefix_id = schedule.get_last(0); prefix_id != -1; prefix_id = schedule.get_next(prefix_id))
        vertex_sets[prefix_id].build_vertex_set(schedule, vertex_sets, &edge[l], r - l, prefix_id);

    l = vertex[v1], r = vertex[v1 + 1];
    for (int prefix_id = schedule.get_last(1); prefix_id != -1; prefix_id = schedule.get_next(prefix_id)) {
        vertex_sets[prefix_id].build_vertex_set(schedule, vertex_sets, &edge[l], r - l, prefix_id);
        if (vertex_sets[prefix_id].get_size() == 0 && prefix_id < schedule.get_basic_prefix_num())
            return 0;
    }

    partial_embedding.push_back(v0);
    partial_embedding.push_back(v1);
    long long ans = 0;
    pattern_matching_aggressive_func(schedule, vertex_sets, partial_embedding, tmp_set, ans, 2, ans_buffer);
    partial_embedding.pop_back();
    partial_embedding.pop_back();
    return ans;
}

void Graph::pattern_matching_aggressive_func(const Schedule_IEP& schedule, VertexSet* vertex_set, VertexSet& subtraction_set, VertexSet& tmp_set, long long& local_ans, int depth, int* ans_buffer)
{
    int loop_set_prefix_id = schedule.get_loop_set_prefix_id(depth);
    int loop_size = vertex_set[loop_set_prefix_id].get_size();
    if (loop_size <= 0)
        return;

    int* loop_data_ptr = vertex_set[loop_set_prefix_id].get_data_ptr();
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
            const VertexSet& vset = vertex_set[loop_set_prefix_id];
            int size_after_restrict = std::lower_bound(vset.get_data_ptr(), vset.get_data_ptr() + vset.get_size(), min_vertex) - vset.get_data_ptr();
            if (size_after_restrict > 0)
                local_ans += VertexSet::unordered_subtraction_size(vertex_set[loop_set_prefix_id], subtraction_set, size_after_restrict);
        }
        else
            local_ans += VertexSet::unordered_subtraction_size(vertex_set[loop_set_prefix_id], subtraction_set);
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
        unsigned int l, r;
        get_edge_index(vertex, l, r);
        bool is_zero = false;
        for (int prefix_id = schedule.get_last(depth); prefix_id != -1; prefix_id = schedule.get_next(prefix_id))
        {
            vertex_set[prefix_id].build_vertex_set(schedule, vertex_set, &edge[l], (int)r - l, prefix_id, vertex);
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
                    unsigned int l, r;
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
            unsigned int l, r;
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
