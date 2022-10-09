#include "../include/labeled_graph.h"
#include "../include/vertex_set.h"
#include "../include/common.h"
#include "../include/motif_generator.h"
#include <cstdio>
#include <sys/time.h>
#include <unistd.h>
#include <cstdlib>
#include <omp.h>
#include <algorithm>
#include <cstring>
#include <mpi.h>
#include <atomic>
#include <vector>
#include <bitset>
#include <set>

int get_pattern_edge_num(const Pattern& p)
{
    int edge_num = 0;
    const int* ptr = p.get_adj_mat_ptr();
    int size = p.get_size();
    for (int i = 0; i < size; ++i)
        for (int j = i + 1; j < size; ++j)
            if (ptr[i * size + j] != 0)
                edge_num += 1;
    return edge_num;
}

//前n个点组成的子图的边数
int get_schedule_edge_num(const Schedule_IEP& p, int n = -1)
{
    if (n == -1)
        n = p.get_size();
    int edge_num = 0;
    const int* ptr = p.get_adj_mat_ptr();
    int size = p.get_size();
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            if (ptr[i * size + j] != 0)
                edge_num += 1;
    return edge_num;
}

bool cmp_pattern_by_edge_num(const Pattern& p1, const Pattern& p2) {
    int num1 = get_pattern_edge_num(p1);
    int num2 = get_pattern_edge_num(p2);
    return (num1 <= num2);
}

//只判断s的子图是否与s0同构，而不考虑s0自同构的问题，只找到一个同构的匹配方案就可以了；如果我们在找到一个frequent pattern后把它所有的pattern都设置成frequent，就可以不用考虑自同构，因为无论记录的是哪个自同构的映射，都是频繁的
bool check_isomorphism_dfs(int schedule_id, const Schedule_IEP& s, const Schedule_IEP& s0, std::vector<int>& mappings_vec, int size, int depth, int* mapping, const int* connected_v = NULL) {
    if (depth - 1 == s0.get_size()) {
        mappings_vec.push_back(schedule_id);
        if (connected_v == NULL)
        {
            for (int i = 0; i < s0.get_size(); ++i)
                mappings_vec.push_back(mapping[i]);
            for (int i = s0.get_size(); i < size; ++i)
                mappings_vec.push_back(-1);
        } else {
            for (int i = 0; i < s0.get_size(); ++i)
                mappings_vec.push_back(connected_v[mapping[i]]);
            for (int i = s0.get_size(); i < size; ++i)
                mappings_vec.push_back(-1);
        }
        return true;
    }
    const int* s_adj = s.get_adj_mat_ptr();
    const int* s0_adj = s0.get_adj_mat_ptr();
    for (int i = 0; i < s0.get_size(); ++i) { //将s0的第depth - 1个点映射为s的第i个点
        bool flag = true;
        for (int j = 0; j < depth - 1; ++j)
            if (mapping[j] == i || s_adj[mapping[j] * s.get_size() + i] != s0_adj[j * s0.get_size() + depth - 1]) {
                flag = false;
                break;
            }
        if (flag) {
            mapping[depth - 1] = i;
            if (check_isomorphism_dfs(schedule_id, s, s0,mappings_vec, size, depth + 1, mapping, connected_v))
                return true;
        }
    }
    return false;
}

bool check_isomorphism(int schedule_id, const Schedule_IEP& s, const Schedule_IEP& s0, std::vector<int>& mappings_vec, int size, const int* connected_v = NULL) {
    int mapping[8]; //默认最大8个点，TODO：不要用magic number
    return check_isomorphism_dfs(schedule_id, s, s0, mappings_vec, size, 1, mapping, connected_v); //搜索全排列
}

int get_connected_v(int* connected_v, const int* adj, int size) { //使用bfs，找到去掉一条边后仍与最后一个点连通的所有点，返回连通点数
    bool* vis = new bool[size];
    memset(vis, 0, size * sizeof(bool));
    int head = 0;
    int tail = 1;
    connected_v[0] = size - 1;
    vis[size - 1] = true;
    while (head < tail) {
        int now = connected_v[head++];
        for (int i = 0; i < size; ++i)
            if (adj[now * size + i] == 1 && vis[i] == false) {
                vis[i] = true;
                connected_v[tail++] = i;
            }
    }
    delete[] vis;
    return tail;
}

//找到s的所有两种子图：1、前depth个点组成的极大子图（depth < s.size）; 2、包含最后一个点且比s少一条边的子图
void generate_mapping_plans(const Schedule_IEP& s, const Schedule_IEP* schedules, std::vector<int>& mappings_vec, int* mapping_start_idx, int& mapping_start_idx_pos, int s_id) {
    for (int depth = 1; depth < s.get_size(); ++depth) {
        mapping_start_idx[mapping_start_idx_pos++] = mappings_vec.size();
        int edge_num = get_schedule_edge_num(s, depth);
        for (int s0_id = 0; s0_id < s_id; ++s0_id) { //已经按照边数排好序了，所以s的子图一定只在s_id之前
            const Schedule_IEP& s0 = schedules[s0_id];
            int edge_num0 = get_schedule_edge_num(s0);
            if (edge_num0 > edge_num)
                break;
            if (edge_num0 != edge_num || s0.get_size() != depth) //当depth < s.size时，只找前depth个点组成的极大子图
                continue;
            if (check_isomorphism(s0_id, s, s0, mappings_vec, s.get_size())) //找到同构，则找下一层（depth+1），否则继续找是否有同构的schedules
                break;
        }
    }
    //depth == s.size
    mapping_start_idx[mapping_start_idx_pos++] = mappings_vec.size();
    int* degree = new int[s.get_size()];
    const int* s_adj = s.get_adj_mat_ptr();
    for (int i = 0; i < s.get_size(); ++i) {
        degree[i] = 0;
        for (int j = 0; j < s.get_size(); ++j)
            if (s_adj[i * s.get_size() + j] == 1)
                ++degree[i];
    }
    //遍历所有比s少一条边且包含s的最后一个点的子图
    int* connected_v = new int[s.get_size()];
    int* tmp_adj = new int[s.get_size() * s.get_size()]; // 用于维护删除一条边吼的s_adj
    int* sub_adj = new int[s.get_size() * s.get_size()]; // 用于新建删除一条边后的schedule（因为连通性可能会变化，所以不能直接用tmp_adj）
    memcpy(tmp_adj, s_adj, s.get_size() * s.get_size() * sizeof(int));
    int s_edge_num = get_schedule_edge_num(s);
    for (int i = 0; i < s.get_size(); ++i)
        for (int j = i + 1; j < s.get_size(); ++j)
            if (s_adj[i * s.get_size() + j] == 1) { //尝试删除(i,j)
                tmp_adj[i * s.get_size() + j] = tmp_adj[j * s.get_size() + i] = 0;
                int connected_v_num = get_connected_v(connected_v, tmp_adj, s.get_size());
                int edge_num = 0;
                for (int tmp_i = 0; tmp_i < connected_v_num; ++tmp_i)
                    for (int tmp_j = 0; tmp_j < connected_v_num; ++tmp_j) {
                        sub_adj[tmp_i * connected_v_num + tmp_j] = tmp_adj[connected_v[tmp_i] * s.get_size() + connected_v[tmp_j]];
                        if (tmp_i < tmp_j && sub_adj[tmp_i * connected_v_num + tmp_j] == 1)
                            ++edge_num;
                    }
                tmp_adj[i * s.get_size() + j] = tmp_adj[j * s.get_size() + i] = 1;
                if (edge_num != s_edge_num - 1) // 只需要查找比s少一条边的子图即可（因为少更多条边的子图一定是某个少一条边的子图的子图）
                    continue;
                Schedule_IEP sub_s(sub_adj, connected_v_num); // TODO：其实这里不需要用schedule，只用pattern就可以了，schedule初始化还要做一些额外操作如IEP，浪费时间
                //edge_num = get_schedule_edge_num(sub_s); //应该edge_num不变，所以这句话删了
                for (int s0_id = 0; s0_id < s_id; ++s0_id) {
                    const Schedule_IEP& s0 = schedules[s0_id];
                    int edge_num0 = get_schedule_edge_num(s0);
                    if (edge_num0 > edge_num)
                        break;
                    if (edge_num0 != edge_num || s0.get_size() != sub_s.get_size())
                        continue;
                    if (check_isomorphism(s0_id, sub_s, s0, mappings_vec, s.get_size(), connected_v))
                        break;
                }
            }
    
    delete[] degree;
    delete[] connected_v;
    delete[] tmp_adj;
    delete[] sub_adj;
}

std::vector<Pattern> generate_fsm_patterns(int max_edge) {
    std::vector<Pattern> res;
    res.clear();
    res.push_back(Pattern(1));

    for (int i = 2; i <= max_edge + 1; ++i) {
        MotifGenerator mg(i);
        std::vector<Pattern> tmp = mg.generate();
        for (const auto& p : tmp) {
            if (get_pattern_edge_num(p) <= max_edge)
                res.push_back(p);
        }
    }
    std::sort(res.begin(), res.end(), cmp_pattern_by_edge_num);
    return res;
}

void LabeledGraph::get_edge_index(int v, int label, unsigned int& l, unsigned int& r) const
{
    int index = v * l_cnt + label;
    l = labeled_vertex[index];
    r = labeled_vertex[index+ 1];
}

//目前不考虑restrict，因为label不同的话可能不存在自同构
//目前不考虑IEP，在算support时只会慢不会错
bool LabeledGraph::get_support_pattern_matching_aggressive_func(const Schedule_IEP& schedule, const char* p_label, VertexSet* vertex_set, VertexSet& subtraction_set, std::vector<std::set<int> >& fsm_set, int depth) const {
    int loop_set_prefix_id = schedule.get_loop_set_prefix_id(depth);
    int loop_size = vertex_set[loop_set_prefix_id].get_size();
    if (loop_size <= 0)
        return false;

    int* loop_data_ptr = vertex_set[loop_set_prefix_id].get_data_ptr();

    if (depth == schedule.get_size() - 1)
    {
        bool match = false;
        for (int i = 0; i < loop_size; ++i)
        {
            int vertex = loop_data_ptr[i];
            if (subtraction_set.has_data(vertex))
                continue;
            match = true;
            fsm_set[depth].insert(vertex);
        }
        return match;
    }

    /*if( depth == schedule.get_size() - schedule.get_in_exclusion_optimize_num() ) {
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
            
    }*/

    //当点加上label后，label不同，自同构就不存在了，所以在这里不考虑限制（自同构的限制在遍历labeled pattern时判断了）
    bool match = false;
    for (int i = 0; i < loop_size; ++i)
    {
        int vertex = loop_data_ptr[i];
        if (subtraction_set.has_data(vertex))
            continue;
        bool is_zero = false;
        for (int prefix_id = schedule.get_last(depth); prefix_id != -1; prefix_id = schedule.get_next(prefix_id))
        {
            unsigned int l, r;
            int target = schedule.get_prefix_target(prefix_id);
            get_edge_index(vertex, p_label[target], l, r);
            vertex_set[prefix_id].build_vertex_set(schedule, vertex_set, &edge[l], (int)r - l, prefix_id, vertex);
            if (vertex_set[prefix_id].get_size() == 0) {
                is_zero = true;
                break;
            }
        }
        if (is_zero) continue;
        //subtraction_set.insert_ans_sort(vertex);
        subtraction_set.push_back(vertex);
        if (get_support_pattern_matching_aggressive_func(schedule, p_label, vertex_set, subtraction_set, fsm_set, depth + 1)) {
            match = true;
            fsm_set[depth].insert(vertex);
        }
        subtraction_set.pop_back();
    }
    return match;
}

//这里返回的不一定是准确的support，只要当前已找到的support>=min_support就直接返回（剪枝）。实际的support>=返回的support。
long long LabeledGraph::get_support_pattern_matching(VertexSet* vertex_set, VertexSet& subtraction_set, const Schedule_IEP& schedule, const char* p_label, std::vector<std::set<int> >& fsm_set, long long min_support) const {
    /*
    //一个点的pattern在omp之前被特殊处理了
    if (schedule.get_size() == 1) {
        long long support = 0;
        for (int vertex = 0; vertex < v_cnt; ++vertex) //TODO: 这里也可以换成一个提前按照v_label排序，会快一些
            if (v_label[vertex] == p_label[0])
                ++support;
        return support;
    }*/
    for (int i = 0; i < schedule.get_size(); ++i)
        fsm_set[i].clear();
    subtraction_set.init();
    //for (int vertex = 0; vertex < v_cnt; ++vertex)
        //if (v_label[vertex] == p_label[0]) //TODO: 这里也可以换成一个提前按照v_label排序，会快一些
    int end_v = label_start_idx[p_label[0] + 1];
    for (int vertex = label_start_idx[p_label[0]]; vertex < end_v; ++vertex)
        {
            bool is_zero = false;
            for (int prefix_id = schedule.get_last(0); prefix_id != -1; prefix_id = schedule.get_next(prefix_id))
            {
                unsigned int l, r;
                int target = schedule.get_prefix_target(prefix_id);
                get_edge_index(vertex, p_label[target], l, r);
                vertex_set[prefix_id].build_vertex_set(schedule, vertex_set, &edge[l], (int)r - l, prefix_id, vertex);
                if (vertex_set[prefix_id].get_size() == 0) {
                    is_zero = true;
                    break;
                }
            }
            if (is_zero)
                continue;
            subtraction_set.push_back(vertex);
            if (get_support_pattern_matching_aggressive_func(schedule, p_label, vertex_set, subtraction_set, fsm_set, 1))
            {
                fsm_set[0].insert(vertex);
                long long support = v_cnt;
                for (int i = 0; i < schedule.get_size(); ++i) {
                    long long count = fsm_set[i].size();
                    if (count < support)
                        support = count;
                }
                if (support >= min_support)
                    return support;
            }
            subtraction_set.pop_back();
            //printf("for %d %d\n", omp_get_thread_num(), vertex);
        }
    long long support = v_cnt;
    for (int i = 0; i < schedule.get_size(); ++i) {
        long long count = fsm_set[i].size();
        if (count < support)
            support = count;
    }
    //if (support > 0)
        //printf("support = %d\n", support);
    return support;
}

void LabeledGraph::get_support_pattern_matching_vertex(int vertex, VertexSet* vertex_set, VertexSet& subtraction_set, const Schedule_IEP& schedule, const char* p_label, std::vector<std::set<int> >& fsm_set, int min_support) const {
    /*
    //一个点的pattern在omp之前被特殊处理了
    if (schedule.get_size() == 1) {
        long long support = 0;
        for (int vertex = 0; vertex < v_cnt; ++vertex) //TODO: 这里也可以换成一个提前按照v_label排序，会快一些
            if (v_label[vertex] == p_label[0])
                ++support;
        return support;
    }*/
    //for (int vertex = 0; vertex < v_cnt; ++vertex)
        //if (v_label[vertex] == p_label[0]) //TODO: 这里也可以换成一个提前按照v_label排序，会快一些
    // int end_v = label_start_idx[p_label[0] + 1];
    // for (int vertex = label_start_idx[p_label[0]]; vertex < end_v; ++vertex)
    //     {
            bool is_zero = false;
            for (int prefix_id = schedule.get_last(0); prefix_id != -1; prefix_id = schedule.get_next(prefix_id))
            {
                unsigned int l, r;
                int target = schedule.get_prefix_target(prefix_id);
                get_edge_index(vertex, p_label[target], l, r);
                vertex_set[prefix_id].build_vertex_set(schedule, vertex_set, &edge[l], (int)r - l, prefix_id, vertex);
                if (vertex_set[prefix_id].get_size() == 0) {
                    is_zero = true;
                    break;
                }
            }
            if (!is_zero) {
                subtraction_set.push_back(vertex);
                if (get_support_pattern_matching_aggressive_func(schedule, p_label, vertex_set, subtraction_set, fsm_set, 1)) {
                    fsm_set[0].insert(vertex);
                }
                subtraction_set.pop_back();
            }
            //printf("for %d %d\n", omp_get_thread_num(), vertex);
        // }
}

void LabeledGraph::traverse_all_labeled_patterns(const Schedule_IEP* schedules, char* all_p_label, char* p_label, const int* mapping_start_idx, const int* mappings, const unsigned int* pattern_is_frequent_index, const unsigned int* is_frequent, int s_id, int depth, int mapping_start_idx_pos, size_t& all_p_label_idx) const { //TODO: 把一些参数转为成员变量试试，这样就能少传一些参数
    const Schedule_IEP& s = schedules[s_id];
    if (depth == s.get_size())
    {
        for (int i = 0; i < s.get_size(); ++i)
            all_p_label[all_p_label_idx++] = p_label[i];
        return;
    }

    int min_label = l_cnt - 1;
    for (int i = s.get_restrict_last(depth); i != -1; i = s.get_restrict_next(i))
        if (min_label > p_label[s.get_restrict_index(i)])
            min_label = p_label[s.get_restrict_index(i)];

    for (int l = 0; l <= min_label; ++l) { //要注意是<=，两个有限制的点可以是同一个label
        p_label[depth] = l;
        bool not_frequent = false;
        for (int mapping_idx = mapping_start_idx[mapping_start_idx_pos]; mapping_idx < mapping_start_idx[mapping_start_idx_pos + 1]; mapping_idx += s.get_size() + 1)
        {
            int s0_id = mappings[mapping_idx];
            const Schedule_IEP& s0 = schedules[s0_id];

            int tmp_p_label[8]; //TODO: 修改这个magic number
            for (int i = 0; i < s0.get_size(); ++i)
                tmp_p_label[i] = p_label[mappings[mapping_idx + 1 + i]];
            unsigned int index = pattern_is_frequent_index[s0_id];
            unsigned int pow = 1;
            for (int i = 0; i < s0.get_size(); ++i) {
                index += tmp_p_label[i] * pow;
                pow *= (unsigned int) l_cnt;
            }
            if ((is_frequent[index >> 5] & ((unsigned int) (1 << (index % 32)))) == 0) {
                not_frequent = true;
                break;
            }
        }
        if (not_frequent)
            continue;
        traverse_all_labeled_patterns(schedules, all_p_label, p_label, mapping_start_idx, mappings, pattern_is_frequent_index, is_frequent, s_id, depth + 1, mapping_start_idx_pos + 1, all_p_label_idx);
    }
}

void LabeledGraph::get_fsm_necessary_info(std::vector<Pattern>& patterns, int max_edge, Schedule_IEP*& schedules, int& schedules_num, int*& mapping_start_idx, int*& mappings, unsigned int*& pattern_is_frequent_index, unsigned int*& is_frequent) const {
    patterns = generate_fsm_patterns(max_edge);
    schedules = (Schedule_IEP*) malloc(sizeof(Schedule_IEP) * patterns.size());
    for (size_t i = 0; i < patterns.size(); ++i) {
        // printf("begin generate schedule %d\n", i);
        // fflush(stdout);
        const Pattern& p = patterns[i];
        bool is_pattern_valid;
        // printf("before generate schedule %d\n", i);
        // printf("v_cnt: %d e_cnt: %d tri_cnt: %lld\n", this->v_cnt, this->e_cnt, this->tri_cnt);
        p.print();
        fflush(stdout);
        new (&schedules[i]) Schedule_IEP(p, is_pattern_valid, 1, 1, 1, this->v_cnt, this->e_cnt, this->tri_cnt);
        schedules[i].print_schedule();
        fflush(stdout);
    }
    printf("after generate schedules\n");
    fflush(stdout);

    printf("pattern num = %ld\n", patterns.size());
    schedules_num = patterns.size();

    std::vector<int> mappings_vec; //TODO: int其实都可以换成char
    mappings_vec.clear();
    mapping_start_idx = new int[patterns.size() * schedules[patterns.size() - 1].get_size() + 1]; //每个schedule的每个点都有一个index，这只是估了一个上界，+1是因为第一位存放对应子图ID
    mapping_start_idx[0] = 0;
    // Check whether the j-th pattern is a subgraph of the i-th pattern.
    int mapping_start_idx_pos = 1;
    for (size_t i = 1; i < patterns.size(); ++i) {
        const Schedule_IEP& s = schedules[i];
        generate_mapping_plans(s, schedules, mappings_vec, mapping_start_idx, mapping_start_idx_pos, i);
    }
    mapping_start_idx[mapping_start_idx_pos] = mappings_vec.size();


    printf("mapping_start_idx_pos = %d   mappings_vec.size() = %lld\n", mapping_start_idx_pos, mappings_vec.size());
    fflush(stdout);

    mappings = new int[mappings_vec.size()];
    memcpy(mappings, &mappings_vec[0], mappings_vec.size() * sizeof(int));
    // check
    mapping_start_idx_pos = 0;
    for (int i = 0; i < patterns.size(); ++i) {
        const Schedule_IEP& s = schedules[i];
        for (int depth = 0; depth < s.get_size(); ++depth)
        {
            for (int mapping_idx = mapping_start_idx[mapping_start_idx_pos]; mapping_idx < mapping_start_idx[mapping_start_idx_pos + 1]; mapping_idx += s.get_size() + 1)
            {
                int s0_id = mappings[mapping_idx];
                const Schedule_IEP& s0 = schedules[s0_id];
                const int* s_adj = s.get_adj_mat_ptr();
                const int* s0_adj = s0.get_adj_mat_ptr();
                bool flag = true;
                int additional_edge_num = 0;
                for (int i = 0; i < s0.get_size(); ++i) {
                    int map_i = mappings[mapping_idx + 1 + i];
                    for (int j = i + 1; j < s0.get_size(); ++j) {
                        int map_j = mappings[mapping_idx + 1 + j];
                        if (s0_adj[i * s0.get_size() + j] != s_adj[map_i * s.get_size() + map_j]) {
                            if (s0_adj[i * s0.get_size() + j] == 0) //可能是s0比s少一条边（在最后一层枚举了删除某条边）
                                additional_edge_num++;
                            else
                                flag = false;
                        }
                    }
                }
                for (int j = mapping_idx; j < mapping_idx + s.get_size() + 1; ++j)
                    printf("%d ", mappings[j]);
                if (flag && additional_edge_num <= 1) {
                    // printf("correct!\n");
                }
                else {
                    printf("wrong!!!!!\n");
                    printf("s_id = %d  s0_id = %d  mapping_idx = %d\n", i, s0_id, mapping_idx);
                    fflush(stdout);
                    //printf("s_id = %d\n", i);
                    //schedules[i].print_schedule();
                    //printf("depth = %d\n", depth);
                    //printf("s0_id = %d\n", s0_id);
                    //schedules[s0_id].print_schedule();
                    //for (int j = mapping_idx; j < mapping_idx + s.get_size() + 1; ++j)
                    //    printf("%d ", mappings[j]);
                    //printf("\n");
                    //fflush(stdout);
                }
            }
            mapping_start_idx_pos++;
        }
    }

    pattern_is_frequent_index = new unsigned int[patterns.size() + 1];
    int index = 0;
    for (int i = 0; i < patterns.size(); ++i) {
        pattern_is_frequent_index[i] = index;
        unsigned int pow = 1;
        for (int j = 0; j < patterns[i].get_size(); ++j)
            pow *= (unsigned int) l_cnt;
        index += pow;
        index = (index + 31) / 32 * 32; //保证pattern_is_frequent_index按照32（4字节）对齐
    }
    pattern_is_frequent_index[patterns.size()] = index;
    is_frequent = new unsigned int[(index + 31) / 32 * 10];
    memset(is_frequent, 0, sizeof(unsigned int) * ((index + 31) / 32 * 10));
}

int LabeledGraph::fsm(int max_edge, long long min_support, int thread_count, double *time_out) {
    std::vector<Pattern> patterns;
    Schedule_IEP* schedules;
    int schedules_num;
    int* mapping_start_idx;
    int* mappings;
    unsigned int* pattern_is_frequent_index; //每个unlabeled pattern对应的所有labeled pattern在is_frequent中的起始位置，按照32对齐（4B）
    unsigned int* is_frequent; //bit vector
    get_fsm_necessary_info(patterns, max_edge, schedules, schedules_num, mapping_start_idx, mappings, pattern_is_frequent_index, is_frequent);

    size_t max_labeled_patterns = 1;
    for (int i = 0; i < max_edge + 1; ++i) //边数最大max_edge，点数最大max_edge + 1
        max_labeled_patterns *= (size_t) l_cnt;
    char* all_p_label = new char[max_labeled_patterns * (max_edge + 1)];
    char* tmp_p_label = new char[max_edge + 1];
    timeval start, end, total_time;
    gettimeofday(&start, NULL);
    long long global_fsm_cnt = 0;
    //特殊处理一个点的pattern
    for (int i = 0; i < l_cnt; ++i)
        if (label_frequency[i] >= min_support) {
            ++global_fsm_cnt;
            is_frequent[i >> 5] |= (unsigned int) (1 << (i % 32));
        }
    if (max_edge != 0)
        global_fsm_cnt = 0;
    int mapping_start_idx_pos = 1;

    for (int i = 1; i < schedules_num; ++i) {
        std::vector<std::vector<int> > automorphisms;
        automorphisms.clear();
        schedules[i].GraphZero_get_automorphisms(automorphisms);
        schedules[i].update_loop_invariant_for_fsm();
        schedules[i].print_schedule();
        size_t all_p_label_idx = 0;
        traverse_all_labeled_patterns(schedules, all_p_label, tmp_p_label, mapping_start_idx, mappings, pattern_is_frequent_index, is_frequent, i, 0, mapping_start_idx_pos, all_p_label_idx);
        size_t job_num = all_p_label_idx / schedules[i].get_size();
        global_fsm_cnt += fsm_pattern_matching(0, job_num, schedules[i], all_p_label, automorphisms, is_frequent, pattern_is_frequent_index[i], max_edge, min_support, thread_count);

        mapping_start_idx_pos += schedules[i].get_size();
        if (get_pattern_edge_num(patterns[i]) != max_edge) //为了使得边数小于max_edge的pattern不被统计。正确性依赖于pattern按照边数排序
            global_fsm_cnt = 0;
        gettimeofday(&end, NULL);
        timersub(&end, &start, &total_time);
        printf("time = %ld.%06ld s.\n", total_time.tv_sec, total_time.tv_usec);
    }

    gettimeofday(&end, NULL);
    timersub(&end, &start, &total_time);

    if(time_out != nullptr) {
        *time_out = double(total_time.tv_sec) + double(total_time.tv_usec) / 1000000;
    }

    fsm_cnt = global_fsm_cnt;
    printf("fsm_cnt = %d\n", fsm_cnt);

    free(schedules);
    delete[] mapping_start_idx;
    delete[] mappings;
    delete[] pattern_is_frequent_index;
    delete[] is_frequent;
    delete[] all_p_label;
    delete[] tmp_p_label;
    
    return fsm_cnt;
}

int LabeledGraph::fsm_pattern_matching(int job_start, int job_end, const Schedule_IEP &schedule, const char *all_p_label, std::vector<std::vector<int> > &automorphisms, unsigned int* is_frequent, unsigned int& pattern_is_frequent_index, int max_edge, int min_support, int thread_count) const {
    long long fsm_cnt = 0;
    #pragma omp parallel num_threads(thread_count) reduction(+: fsm_cnt)
    {
        VertexSet* vertex_set = new VertexSet[schedule.get_total_prefix_num()];
        VertexSet subtraction_set;
        std::vector<std::set<int> > fsm_set;
        fsm_set.clear();
        long long local_fsm_cnt = 0;
        for (int j = 0; j < max_edge + 1; ++j) { //至多max_edge + 1个点
            fsm_set.push_back(std::set<int>());
            fsm_set.back().clear();
        }
        char* p_label = new char[max_edge + 1];
        #pragma omp for schedule(dynamic) nowait
        for (size_t job_id = job_start; job_id < job_end; ++job_id) {
            size_t job_start_idx = job_id * schedule.get_size();
            for (int j = 0; j < schedule.get_size(); ++j)
                p_label[j] = all_p_label[job_start_idx + j];
            long long support;
            support = get_support_pattern_matching(vertex_set, subtraction_set, schedule, p_label, fsm_set, min_support);
            if (support >= min_support) {
                local_fsm_cnt++;
                //这里本来是单线程时输出用的
                /*printf("support = %d   frequent pattern's label = ", support);
                for (int j = 0; j < schedule.get_size(); ++j)
                    for (auto it = label_map.begin(); it != label_map.end(); ++it)
                        if (it->second == p_label[j]) {
                            printf("%d ", it->first);
                            break;
                        }
                printf("   (");
                for (int j = 0; j < schedule.get_size(); ++j)
                    printf("%d ", p_label[j]);
                printf(")\n");*/
                // printf("cpu support: %lld job_id:%d-%d-%d\n", support, job_start, job_id, job_end);
                    
                char tmp_p_label[8]; //TODO: 修改这个magic number
                for (const auto& aut : automorphisms) { //遍历所有自同构，为自己和所有自同构的is_frequent赋值
                    for (int j = 0; j < schedule.get_size(); ++j)
                        tmp_p_label[j] = p_label[aut[j]];
                    unsigned int index = pattern_is_frequent_index;
                    unsigned int pow = 1;
                    for (int j = 0; j < schedule.get_size(); ++j) {
                        index += tmp_p_label[j] * pow;
                        pow *= (unsigned int) l_cnt;
                    }
                    #pragma omp critical
                    {
                        is_frequent[index >> 5] |= (unsigned int) (1 << (index % 32));
                    }
                }
            }
            // if(job_id % 100 == 0) {
                // #pragma omp critical
                // {
                    // printf("finish job_id: %d/%d\n", job_id, job_end);
                // }
            // }
        }
        delete[] vertex_set;
        delete[] p_label;
        fsm_cnt += local_fsm_cnt;
    }
    return fsm_cnt;
}

int LabeledGraph::fsm_vertex(int max_edge, long long min_support, int thread_count, double *time_out) {
    std::vector<Pattern> patterns;
    Schedule_IEP* schedules;
    int schedules_num;
    int* mapping_start_idx;
    int* mappings;
    unsigned int* pattern_is_frequent_index; //每个unlabeled pattern对应的所有labeled pattern在is_frequent中的起始位置，按照32对齐（4B）
    unsigned int* is_frequent; //bit vector
    get_fsm_necessary_info(patterns, max_edge, schedules, schedules_num, mapping_start_idx, mappings, pattern_is_frequent_index, is_frequent);

    size_t max_labeled_patterns = 1;
    for (int i = 0; i < max_edge + 1; ++i) //边数最大max_edge，点数最大max_edge + 1
        max_labeled_patterns *= (size_t) l_cnt;
    char* all_p_label = new char[max_labeled_patterns * (max_edge + 1)];
    char* tmp_p_label = new char[max_edge + 1];
    timeval start, end, total_time;
    gettimeofday(&start, NULL);
    long long global_fsm_cnt = 0;
    //特殊处理一个点的pattern
    for (int i = 0; i < l_cnt; ++i)
        if (label_frequency[i] >= min_support) {
            ++global_fsm_cnt;
            is_frequent[i >> 5] |= (unsigned int) (1 << (i % 32));
        }
    if (max_edge != 0)
        global_fsm_cnt = 0;
    int mapping_start_idx_pos = 1;

    for (int i = 1; i < schedules_num; ++i) {
        std::vector<std::vector<int> > automorphisms;
        automorphisms.clear();
        schedules[i].GraphZero_get_automorphisms(automorphisms);
        schedules[i].update_loop_invariant_for_fsm();
        schedules[i].print_schedule();
        size_t all_p_label_idx = 0;
        traverse_all_labeled_patterns(schedules, all_p_label, tmp_p_label, mapping_start_idx, mappings, pattern_is_frequent_index, is_frequent, i, 0, mapping_start_idx_pos, all_p_label_idx);

        size_t job_num = all_p_label_idx / schedules[i].get_size();
        for (size_t job_id = 0; job_id < job_num; ++job_id) {
           global_fsm_cnt += fsm_pattern_matching_vertex(job_id, schedules[i], all_p_label, automorphisms, is_frequent, pattern_is_frequent_index[i] ,max_edge, min_support, thread_count);
        }
        mapping_start_idx_pos += schedules[i].get_size();
        if (get_pattern_edge_num(patterns[i]) != max_edge) //为了使得边数小于max_edge的pattern不被统计。正确性依赖于pattern按照边数排序
            global_fsm_cnt = 0;
        gettimeofday(&end, NULL);
        timersub(&end, &start, &total_time);
        printf("time = %ld.%06ld s.\n", total_time.tv_sec, total_time.tv_usec);
    }

    gettimeofday(&end, NULL);
    timersub(&end, &start, &total_time);

    if(time_out != nullptr) {
        *time_out = double(total_time.tv_sec) + double(total_time.tv_usec) / 1000000;
    }

    fsm_cnt = global_fsm_cnt;
    printf("fsm_cnt = %d\n", fsm_cnt);

    free(schedules);
    delete[] mapping_start_idx;
    delete[] mappings;
    delete[] pattern_is_frequent_index;
    delete[] is_frequent;
    delete[] all_p_label;
    delete[] tmp_p_label;
    
    return fsm_cnt;
}

int LabeledGraph::fsm_pattern_matching_vertex(int job_id, const Schedule_IEP &schedule, const char *all_p_label, std::vector<std::vector<int> > &automorphisms, unsigned int* is_frequent, unsigned int& pattern_is_frequent_index, int max_edge, int min_support, int thread_count) const {
    
    int fsm_cnt = 0;
    size_t job_start_idx = job_id * schedule.get_size();
    char* p_label = new char[max_edge + 1];
    for (int j = 0; j < schedule.get_size(); ++j)
        p_label[j] = all_p_label[job_start_idx + j];
    std::vector<std::set<int> > fsm_set;
    fsm_set.clear();
    for (int j = 0; j < max_edge + 1; ++j) { //至多max_edge + 1个点
        fsm_set.push_back(std::set<int>());
        fsm_set.back().clear();
    }
    #pragma omp parallel num_threads(thread_count)
    {
        VertexSet* vertex_set = new VertexSet[schedule.get_total_prefix_num()];
        std::vector<std::set<int> > local_fsm_set;
        local_fsm_set.clear();
        for (int j = 0; j < max_edge + 1; ++j) { //至多max_edge + 1个点
            local_fsm_set.push_back(std::set<int>());
            local_fsm_set.back().clear();
        }
        VertexSet subtraction_set;
        subtraction_set.init();
        // int ans = 0;
        #pragma omp for schedule(dynamic, 10) nowait
        for (int vertex = label_start_idx[p_label[0]]; vertex < label_start_idx[p_label[0] + 1]; ++vertex) {
            int support = v_cnt;
            for (int i = 0; i < schedule.get_size(); ++i) {
                int count = fsm_set[i].size();
                if (count < support)
                    support = count;
            }
            if (support < min_support) {
                get_support_pattern_matching_vertex(vertex, vertex_set, subtraction_set, schedule, p_label, local_fsm_set, min_support);
                double t1 = get_wall_time();
                for(int j = 0; j < max_edge + 1; ++j) {
                    for(auto v : local_fsm_set[j]) {
                        #pragma omp critical
                        {
                            fsm_set[j].insert(v);
                        }
                    }
                    local_fsm_set[j].clear();
                }
                double t2 = get_wall_time();
            }
        }
        // for(int j = 0; j < max_edge + 1; ++j) {
        //     for(auto v : local_fsm_set[j]) {
        //         #pragma omp critical
        //         {
        //             fsm_set[j].insert(v);
        //         }
        //     }
        //     local_fsm_set[j].clear();
        // }
        // delete[] vertex_set;
        // printf("finished: thread %d point_count:%d %lf\n", omp_get_thread_num(), ans, t2 - t1);
        // fflush(stdout);
    }
    int support = v_cnt;
    for (int i = 0; i < schedule.get_size(); ++i) {
        int count = fsm_set[i].size();
        // printf("fsm_set[%d]: %d\n",i, count);
        if (count < support)
            support = count;
    }
    if(job_id % 100 == 0) {
        printf("job_id:%d support: %d\n", job_id, support);
    }
    if (support >= min_support) {
        fsm_cnt++;
        char tmp_p_label[8]; //TODO: 修改这个magic number
        for (const auto& aut : automorphisms) { //遍历所有自同构，为自己和所有自同构的is_frequent赋值
            for (int j = 0; j < schedule.get_size(); ++j)
                tmp_p_label[j] = p_label[aut[j]];
            unsigned int index = pattern_is_frequent_index;
            unsigned int pow = 1;
            for (int j = 0; j < schedule.get_size(); ++j) {
                index += tmp_p_label[j] * pow;
                pow *= (unsigned int) l_cnt;
            }
            is_frequent[index >> 5] |= (unsigned int) (1 << (index % 32));
        }
    }
    delete[] p_label;
    return fsm_cnt;
}