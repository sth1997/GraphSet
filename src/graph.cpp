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

int Graph::intersection_size(int v1,int v2) {
    unsigned int l1, r1;
    this->get_edge_index(v1, l1, r1);
    unsigned int l2, r2;
    this->get_edge_index(v2, l2, r2);
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
    this->get_edge_index(v1, l1, r1);
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
    this->get_edge_index(v1, l1, r1);
    unsigned int l2, r2;
    this->get_edge_index(v2, l2, r2);
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
        this->get_edge_index(v, l, r);
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
        this->get_edge_index(v, l, r);
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
            this->get_edge_index(v, l, r);
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
    // assert(!(is_local_graph));
    l = vertex[v];
    r = vertex[v + 1];
}

void Graph::pattern_matching_func(const Schedule& schedule, VertexSet* vertex_set, VertexSet& subtraction_set, long long& local_ans, int depth, bool clique)
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
            local_ans += VertexSet::unorderd_subtraction_size(vertex_set[loop_set_prefix_id], subtraction_set);
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
        this->get_edge_index(vertex, l, r);
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

// static double local_time;

long long Graph::pattern_matching(const Schedule& schedule, int thread_count, bool clique)
{
//    intersection_times_low = intersection_times_high = 0;
//    dep1_cnt = dep2_cnt = dep3_cnt = 0;
    // local_time = 0;
    long long global_ans = 0;
#pragma omp parallel num_threads(thread_count) reduction(+: global_ans)
    {
        Bitmap *bs = new Bitmap(v_cnt);
     //   double start_time = get_wall_time();
     //   double current_time;
        VertexSet* vertex_set = new VertexSet[schedule.get_total_prefix_num()];
        VertexSet subtraction_set;
        VertexSet tmp_set;
        if(!clique)
            subtraction_set.init();
        long long local_ans = 0;
        // TODO : try different chunksize
 #pragma omp for schedule(dynamic, 1) nowait
        for (int vertex = 0; vertex < v_cnt; ++vertex)
        {
            bs->set_0();
            unsigned int l, r;
            this->get_edge_index(vertex, l, r);
            // assert(bs->count() == 0);
            for (int prefix_id = schedule.get_last(0); prefix_id != -1; prefix_id = schedule.get_next(prefix_id))
            {
                vertex_set[prefix_id].build_vertex_set(schedule, vertex_set, bs, &edge[l], (int)r - l, prefix_id, -1, clique);
                // if(!(bs->count() == vertex_set[prefix_id].get_size())){
                //     #pragma omp critical
                //     {
                //         printf("tid:%d vertex:%d size:%d %d %lld\n",omp_get_thread_num(), vertex, r - l, bs->count(), vertex_set[prefix_id].get_size());
                //         for(int i = 0; i < vertex_set[prefix_id].get_size(); i ++){
                //             printf("%d ",vertex_set[prefix_id].get_data(i));
                //         }
                //         printf("\n");
                //         assert(false);
                //     }
                // }
            }
            if(!clique)
                subtraction_set.push_back(vertex);
            if(true){
                if(clique)
                    clique_matching_func(schedule, vertex_set, bs, local_ans, 1);
                else 
                    pattern_matching_aggressive_func(schedule, vertex_set, subtraction_set, tmp_set, local_ans, 1);
            }
            else
                pattern_matching_func(schedule, vertex_set, subtraction_set, local_ans, 1, clique);
            if(!clique)
                subtraction_set.pop_back();
            //printf("for %d %d\n", omp_get_thread_num(), vertex);
        }
        //double end_time = get_wall_time();
        //printf("my thread time %d %.6lf\n", omp_get_thread_num(), end_time - start_time);
        delete[] vertex_set;
        delete bs;
        // TODO : Computing multiplicty for a pattern
        global_ans += local_ans;
        //printf("local_ans %d %lld\n", omp_get_thread_num(), local_ans);
        
    }
    // printf("local_time %.3lf\n", local_time);
    return global_ans / schedule.get_in_exclusion_optimize_redundancy();
}


// void Graph::use_local_graph(const Schedule& schedule, VertexSet* vertex_set, VertexSet& subtraction_set, VertexSet& tmp_set, long long &local_ans, int depth, int u, int v, int *loop_data_ptr, int loop_size){

//     // is_local_graph = true;

//     // this->pattern_matching_aggressive_func(schedule, vertex_set, subtraction_set, tmp_set, local_ans, depth);
//     // return;
//     // before we enter the third level
    
//     // if the Graph is DAG here

//     double t_1, t_2;

//     assert(u > v);

//     t_1 = get_wall_time();

//     int new_u,new_v, new_loop;

//     Local_Graph * lg = new Local_Graph(*this, u, v, loop_data_ptr, loop_size, new_loop);


//     // pop u & v
//     subtraction_set.pop_back();
//     subtraction_set.pop_back();
//     // push 0 & 1
//     subtraction_set.push_back(new_loop + 1);
//     subtraction_set.push_back(new_loop);

//     // deal with vertex_set
//     // 理论上来说，改这个应该不会有影响

//     // int *new0 = new int [loop_size * 2];
//     int *new1 = new int [new_loop];

//     // int *tmp0 = new int [vertex_set[0].get_size() + 5], size0 = vertex_set[0].get_size();
//     int *tmp1 = new int [vertex_set[1].get_size() + 5], size1 = vertex_set[1].get_size();


//     // printf("%d: %d %d\n", 1, , loop_size);
//     memcpy(tmp1, vertex_set[1].get_data_ptr(), sizeof(int) * (size1));

//     for(int i = 0; i < new_loop; i++){
//         new1[i] = i;
//     }

//     vertex_set[1].copy(new_loop, new1);


//     t_2 = get_wall_time();

//     // #pragma omp critical
//     // {
//         // local_time += (t_2 - t_1);
//     // }


//     lg->pattern_matching_aggressive_func(schedule, vertex_set, subtraction_set, tmp_set, local_ans, depth);
    
//     t_1 = get_wall_time();

//     // restore vertex_set
//     vertex_set[1].copy((size1), tmp1);


//     subtraction_set.pop_back();
//     subtraction_set.pop_back();
//     // push 0 & 1
//     subtraction_set.push_back(u);
//     subtraction_set.push_back(v);

//     delete lg;
//     // delete[] tmp0;
//     delete[] tmp1;
//     // delete[] new0;
//     delete[] new1;

//     t_2 = get_wall_time();

//     // #pragma omp critical
//     // {
//         // local_ans += (t_2 - t_1);
//     // }
// }


void Graph::clique_matching_func(const Schedule& schedule, VertexSet* vertex_set, Bitmap* bs, long long& local_ans, int depth) {
    int loop_set_prefix_id = schedule.get_loop_set_prefix_id(depth);
    int loop_size = vertex_set[loop_set_prefix_id].get_size();
    int* loop_data_ptr = vertex_set[loop_set_prefix_id].get_data_ptr();

    // if (depth == schedule.get_size() - 1)
    // {
    //     // local_ans += vertex_set[loop_set_prefix_id].get_size();
    //     return;
    // }


    for (int i = 0; i < loop_size; ++i) {
        int vertex = loop_data_ptr[i];
        unsigned int l, r;
        this->get_edge_index(vertex, l, r);
        int prefix_id = schedule.get_last(depth);// only one prefix
        if(depth == schedule.get_size() - 2)
            vertex_set[prefix_id].build_vertex_set_only_size(schedule, vertex_set, bs, &edge[l], (int)r - l, prefix_id, -1, false);
        else 
            vertex_set[prefix_id].build_vertex_set(schedule, vertex_set, bs, &edge[l], (int)r - l, prefix_id, -1, false);

        if(depth == schedule.get_size() - 2)
            local_ans += vertex_set[prefix_id].get_size();
        else {
            if(vertex_set[prefix_id].get_size() > 0) {
                clique_matching_func(schedule, vertex_set, bs, local_ans, depth + 1);
            }
            int *_data = vertex_set[prefix_id].get_data_ptr(), _size = vertex_set[prefix_id].get_size();
            for(int j = 0; j < loop_size; j++) {
                // set1
                bs->flip_bit(loop_data_ptr[j]);
            }
            for(int j = 0; j < _size; j++) {
                // set1 and set2
                bs->flip_bit(_data[j]);
            }
            // assert(bs->count() == loop_size);
        }
    }
}

void Graph::pattern_matching_aggressive_func(const Schedule& schedule, VertexSet* vertex_set, VertexSet& subtraction_set, VertexSet& tmp_set, long long& local_ans, int depth)
{
    int loop_set_prefix_id = schedule.get_loop_set_prefix_id(depth);

    // assert(depth == loop_set_prefix_id + 1);
    int loop_size = vertex_set[loop_set_prefix_id].get_size();
    if (loop_size <= 0)
        return;


    int* loop_data_ptr = vertex_set[loop_set_prefix_id].get_data_ptr();


    // if(depth == 2 && is_local_graph == false && is_clique_optimzition) {
    //     // printf("use local_graph.");
    //     int u =  subtraction_set.get_data(0), v = subtraction_set.get_data(1);
    //     use_local_graph(schedule, vertex_set, subtraction_set, tmp_set, local_ans, depth, u, v, loop_data_ptr, loop_size);
    //     return;
    // }
    //Case: in_exclusion_optimize_num > 1
    if( depth == schedule.get_size() - schedule.get_in_exclusion_optimize_num() ) {
        // assert(false);
        int in_exclusion_optimize_num = schedule.get_in_exclusion_optimize_num();
        int loop_set_prefix_ids[ in_exclusion_optimize_num ];
        loop_set_prefix_ids[0] = loop_set_prefix_id;
        for(int i = 1; i < in_exclusion_optimize_num; ++i)
            loop_set_prefix_ids[i] = schedule.get_loop_set_prefix_id( depth + i );
        for(int optimize_rank = 0; optimize_rank < schedule.in_exclusion_optimize_group.size(); ++optimize_rank) {
            const std::vector< std::vector<int> >& cur_graph = schedule.in_exclusion_optimize_group[optimize_rank];
            long long val = schedule.in_exclusion_optimize_val[optimize_rank];
            for(int cur_graph_rank = 0; cur_graph_rank < cur_graph.size(); ++ cur_graph_rank) {
//                VertexSet tmp_set;
                
                //if size == 1 , we will not call intersection(...)
                //so we will not allocate memory for data
                //otherwise, we need to copy the data to do intersection(...)
                if(cur_graph[cur_graph_rank].size() == 1) {
                    int id = loop_set_prefix_ids[cur_graph[cur_graph_rank][0]];
                    val = val * VertexSet::unorderd_subtraction_size(vertex_set[id], subtraction_set);
                }
                else {
                    int id = loop_set_prefix_ids[cur_graph[cur_graph_rank][0]];
                    tmp_set.copy(vertex_set[id].get_size(), vertex_set[id].get_data_ptr());

                    for(int i = 1; i < cur_graph[cur_graph_rank].size(); ++i) {
                        int id = loop_set_prefix_ids[cur_graph[cur_graph_rank][i]];
                        tmp_set.intersection_with(vertex_set[id]);
                    }
                    val = val * VertexSet::unorderd_subtraction_size(tmp_set, subtraction_set);
                }
                if( val == 0 ) break;

            }
            local_ans += val;
        }
        return;
            
    }
    // if(depth >= 3 && is_local_graph) {
    //     printf("depth:%d size:%d\n", depth, loop_size);
    // }
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
                local_ans += VertexSet::unorderd_subtraction_size(vertex_set[loop_set_prefix_id], subtraction_set, size_after_restrict);
        }
        else
            local_ans += VertexSet::unorderd_subtraction_size(vertex_set[loop_set_prefix_id], subtraction_set);
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
        unsigned int l, r;
        this->get_edge_index(vertex, l, r);
        bool is_zero = false;
        for (int prefix_id = schedule.get_last(depth); prefix_id != -1; prefix_id = schedule.get_next(prefix_id))
        {
            vertex_set[prefix_id].build_vertex_set(schedule, vertex_set, &edge[l], (int)r - l, prefix_id, vertex);
            if( vertex_set[prefix_id].get_size() == 0) {
                is_zero = true;
                break;
            }
        }
        if( is_zero ) continue;
        //subtraction_set.insert_ans_sort(vertex);
        subtraction_set.push_back(vertex);
        pattern_matching_aggressive_func(schedule, vertex_set, subtraction_set, tmp_set, local_ans, depth + 1);
        subtraction_set.pop_back();
    }
}

long long Graph::pattern_matching_mpi(const Schedule& schedule, int thread_count, bool clique)
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
                pattern_matching_aggressive_func(schedule, vertex_set, subtraction_set, tmp_set, local_ans, 1);
                subtraction_set.pop_back();
            };
            for (std::pair<int, int> range;;)
            {
                if ((range = gm.get_vertex_range()).first == -1) break;
                //for (int vertex = v_cnt - range.second; vertex < v_cnt - range.first; vertex++) {//backwards slower than forwards
                for (int vertex = range.first; vertex < range.second; vertex++) {
                    unsigned int l, r;
                    this->get_edge_index(vertex, l, r);
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

void Graph::pattern_matching_aggressive_func_mpi(const Schedule& schedule, VertexSet* vertex_set, VertexSet& subtraction_set, VertexSet &tmp_set, long long& local_ans, int depth)
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
                local_ans += VertexSet::unorderd_subtraction_size(vertex_set[loop_set_prefix_id], subtraction_set, size_after_restrict);
        }
        else
            local_ans += VertexSet::unorderd_subtraction_size(vertex_set[loop_set_prefix_id], subtraction_set);
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
            this->get_edge_index(vertex, l, r);
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
        pattern_matching_aggressive_func(schedule, vertex_set, subtraction_set, tmp_set, local_ans, depth + 1);
        subtraction_set.pop_back();
    }
}
