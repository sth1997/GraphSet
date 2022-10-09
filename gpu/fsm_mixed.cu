#undef NDEBUG
#include <graph.h>
#include <dataloader.h>
#include <vertex_set.h>
#include <common.h>
#include <schedule_IEP.h>

#include <cassert>
#include <cstring>
#include <cstdint>
#include <string>
#include <algorithm>

#include <sys/time.h>
#include <omp.h>

#include "gpu_fsm_vertex_set.cuh"
#include "gpu_schedule.cuh"
#include "gpu_const.cuh"
#include "timeinterval.h"
#include "utils.cuh"
#include "gpu_bitvector.cuh"


__device__ unsigned long long dev_sum = 0;
__device__ unsigned int dev_cur_labeled_pattern = 0;


TimeInterval allTime;
TimeInterval tmpTime;



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

constexpr int MAX_DEPTH = 5; // 非递归pattern matching支持的最大深度

template <int depth>
__device__ bool GPU_pattern_matching_func(const GPUSchedule* schedule, GPUVertexSet* vertex_set, GPUVertexSet& subtraction_set,
    uint32_t *edge, uint32_t* labeled_vertex, const char* p_label, GPUBitVector* fsm_set, int l_cnt)
{

    int loop_set_prefix_id = schedule->get_loop_set_prefix_id(depth);
    int loop_size = vertex_set[loop_set_prefix_id].get_size();

    uint32_t* loop_data_ptr = vertex_set[loop_set_prefix_id].get_data_ptr();

    bool local_match = false;
    __shared__ bool block_match[WARPS_PER_BLOCK];
    if (depth == schedule->get_size() - 1) {
        if (threadIdx.x % THREADS_PER_WARP == 0) { //TODO: 改成并行，不过要注意现在fsm_set不支持并行
            for (int i = 0; i < loop_size; ++i)
            {
                int vertex = loop_data_ptr[i];
                if (subtraction_set.has_data_size(vertex, depth))
                    continue;
                local_match = true;
                fsm_set[depth].insert_and_update(vertex);
            }
            block_match[threadIdx.x / THREADS_PER_WARP] = local_match;
        }
        __threadfence_block();
        return block_match[threadIdx.x / THREADS_PER_WARP]; 
    }

    for (int i = 0; i < loop_size; ++i)
    {
        // if(depth == 1 && threadIdx.x % THREADS_PER_WARP == 0 && i % 100 == 0) {
        //     printf("i:%d\n", i);
        // }
        uint32_t v = loop_data_ptr[i];
        if (subtraction_set.has_data_size(v, depth))
            continue;
        bool is_zero = false;
        for (int prefix_id = schedule->get_last(depth); prefix_id != -1; prefix_id = schedule->get_next(prefix_id))
        {
            unsigned int l, r;
            int target = schedule->get_prefix_target(prefix_id);
            get_labeled_edge_index(v, p_label[target], l, r);
            vertex_set[prefix_id].build_vertex_set(schedule, vertex_set, &edge[l], r - l, prefix_id);
            if (vertex_set[prefix_id].get_size() == 0) {
                is_zero = true;
                break;
            }
        }
        if (is_zero)
            continue;
        if (threadIdx.x % THREADS_PER_WARP == 0)
            subtraction_set.put(v, depth);
        __threadfence_block();

        if (GPU_pattern_matching_func<depth + 1>(schedule, vertex_set, subtraction_set, edge, labeled_vertex, p_label, fsm_set, l_cnt)) {
            local_match = true;
            if (threadIdx.x % THREADS_PER_WARP == 0)
                fsm_set[depth].insert_and_update(v);
        }
        // if (threadIdx.x % THREADS_PER_WARP == 0)
        //     subtraction_set.pop_back();
        __threadfence_block();
    }
    // if(threadIdx.x % THREADS_PER_WARP == 0 && depth == 3)
    //     printf("\n");
    return local_match;
}

    template <>
__device__ bool GPU_pattern_matching_func<MAX_DEPTH>(const GPUSchedule* schedule, GPUVertexSet* vertex_set, GPUVertexSet& subtraction_set,
    uint32_t *edge, uint32_t* labeled_vertex, const char* p_label, GPUBitVector* fsm_set, int l_cnt)
{
    // assert(false);
}

/**
 * @note `buffer_size`实际上是每个节点的最大邻居数量，而非所用空间大小
 */
// job_id range from [job_start, end)
__global__ void gpu_pattern_matching(uint32_t job_start, uint32_t job_end, uint32_t v_cnt, uint32_t buffer_size, uint32_t *edge, uint32_t* labeled_vertex, int* v_label, uint32_t* tmp, const GPUSchedule* schedule, char* all_p_label, GPUBitVector* global_fsm_set, int* automorphisms, unsigned int* is_frequent, unsigned int* label_start_idx, int automorphisms_cnt, long long min_support, unsigned int pattern_is_frequent_index, int l_cnt) {

    __shared__ unsigned int block_pattern_idx[WARPS_PER_BLOCK];
    __shared__ bool block_break_flag[WARPS_PER_BLOCK];
    //之后考虑把tmp buffer都放到shared里来（如果放得下）
    extern __shared__ GPUVertexSet block_vertex_set[];
    
    int num_prefixes = schedule->get_total_prefix_num();
    int num_vertex_sets_per_warp = num_prefixes + 2;

    int wid = threadIdx.x / THREADS_PER_WARP; // warp id within the block
    int lid = threadIdx.x % THREADS_PER_WARP; // lane id
    int global_wid = blockIdx.x * WARPS_PER_BLOCK + wid; // global warp id
    unsigned int &pattern_idx = block_pattern_idx[wid];
    GPUVertexSet *vertex_set = block_vertex_set + wid * num_vertex_sets_per_warp;
    char* p_label = ((char*) (block_vertex_set)) + schedule->p_label_offset + (schedule->max_edge + 1) * wid;

    //__shared__ GPUBitVector* block_fsm_set[WARPS_PER_BLOCK];
    //GPUBitVector*& fsm_set = block_fsm_set[wid];
    GPUBitVector* fsm_set = global_fsm_set + global_wid * schedule->get_size();


    if (lid == 0) {
        pattern_idx = 0;
        uint32_t offset = buffer_size * global_wid * num_vertex_sets_per_warp;
        for (int i = 0; i < num_vertex_sets_per_warp; ++i)
        {
            vertex_set[i].set_data_ptr(tmp + offset); // 注意这是个指针+整数运算，自带*4
            offset += buffer_size;
        }
    }
    GPUVertexSet& subtraction_set = vertex_set[num_prefixes];
    //GPUVertexSet& tmp_set = vertex_set[num_prefixes + 1];

    __threadfence_block(); //之后考虑把所有的syncthreads都改成syncwarp

    while (true) {
        if (lid == 0) {
            //if(++edgeI >= edgeEnd) { //这个if语句应该是每次都会发生吧？（是的
                pattern_idx = atomicAdd(&dev_cur_labeled_pattern, 1); //每个warp负责一个pattern，而不是负责一个点或一条边
                //edgeEnd = min(edge_num, edgeI + 1); //这里不需要原子读吗
                unsigned int job_id = pattern_idx;
                if (job_id < job_end)
                {
                    subtraction_set.init();
                    //subtraction_set.push_back(edge_from[i]);
                    //subtraction_set.push_back(edge[i]);
                    size_t job_start_idx = job_id * schedule->get_size();
                    for (int j = 0; j < schedule->get_size(); ++j)
                        p_label[j] = all_p_label[job_start_idx + j];
                }
            //}
        }

        __threadfence_block();

        unsigned int job_id = pattern_idx;
        if(job_id >= job_end) break;

        if (lid < schedule->get_size())
            fsm_set[lid].clear();
        __threadfence_block();
        
        int end_v = label_start_idx[p_label[0] + 1];
        block_break_flag[wid] = false;
        for (int vertex = label_start_idx[p_label[0]]; vertex < end_v; ++vertex) {
            bool is_zero = false;
            for (int prefix_id = schedule->get_last(0); prefix_id != -1; prefix_id = schedule->get_next(prefix_id)) {
                unsigned int l, r;
                int target = schedule->get_prefix_target(prefix_id);
                get_labeled_edge_index(vertex, p_label[target], l, r);
                vertex_set[prefix_id].build_vertex_set(schedule, vertex_set, &edge[l], (int)r - l, prefix_id);
                if (vertex_set[prefix_id].get_size() == 0) {
                    is_zero = true;
                    break;
                }
            }
            if (is_zero)
                continue;
            if (lid == 0)
                subtraction_set.put(vertex, 0);
            __threadfence_block();

            if (GPU_pattern_matching_func<1>(schedule, vertex_set, subtraction_set, edge, labeled_vertex, p_label, fsm_set, l_cnt)) {
                if (lid == 0) //TODO: 目前insert只让0号线程执行，之后考虑32个线程同时执行，看会不会出错（好像是不会）
                {
                    fsm_set[0].insert_and_update(vertex);
                    __threadfence_block();
                    long long support = v_cnt;
                    for (int i = 0; i < schedule->get_size(); ++i) {
                        long long count = fsm_set[i].get_non_zero_cnt();
                        if (count < support)
                            support = count;
                    }
                    if(lid == 0) {
                        // printf("%d\n", support);
                        if (support >= min_support) {
                            // printf("gpu support: %lld job_id:%d-%d-%d\n", support, job_start, job_id, job_end);
                            block_break_flag[wid] =true;
                            atomicAdd(&dev_sum, 1);
                            for (int aut_id = 0; aut_id < automorphisms_cnt; ++aut_id) { //遍历所有自同构，为自己和所有自同构的is_frequent赋值
                                int* aut = automorphisms + aut_id * schedule->get_size();
                                unsigned int index = pattern_is_frequent_index;
                                unsigned int pow = 1;
                                for (int j = 0; j < schedule->get_size(); ++j) {
                                    index += p_label[aut[j]] * pow;
                                    pow *= (unsigned int) l_cnt;
                                }
                                atomicOr(&is_frequent[index >> 5], (unsigned int) (1 << (index % 32)));
                            }
                            __threadfence_block();
                        }
                    }
                }
            }
            if (block_break_flag[wid] == true)
                break;
            if (lid == 0)
                subtraction_set.pop_back();
        }
        __threadfence_block();
    }
}

long long pattern_matching_init(const LabeledGraph *g, const Schedule_IEP& schedule, const std::vector<std::vector<int> >& automorphisms, unsigned int pattern_is_frequent_index, unsigned int* dev_is_frequent, uint32_t* dev_edge, uint32_t* dev_labeled_vertex, int* dev_v_label, uint32_t* dev_tmp, int max_edge, int job_start, int job_end, char* dev_all_p_label, GPUBitVector* fsm_set, uint32_t* dev_label_start_idx, long long min_support) {

    printf("total prefix %d\n", schedule.get_total_prefix_num());

    schedule.print_schedule();
    tmpTime.check(); 

    unsigned long long sum = 0; //sum是这个pattern的所有labeled pattern中频繁的个数
    gpuErrchk( cudaMemcpyToSymbol(dev_sum, &sum, sizeof(sum)));

    
    int* dev_automorphisms;
    int* host_automorphisms = new int[schedule.get_size() * automorphisms.size()];
    {
        int tmp_idx = 0;
        for (const auto& aut : automorphisms) {
            for (int i = 0; i < schedule.get_size(); ++i)
                host_automorphisms[tmp_idx++] = aut[i];
        }
    }
    gpuErrchk( cudaMalloc((void**)&dev_automorphisms, sizeof(int) * schedule.get_size() * automorphisms.size()));
    gpuErrchk( cudaMemcpy(dev_automorphisms, host_automorphisms, sizeof(int) * schedule.get_size() * automorphisms.size(), cudaMemcpyHostToDevice));

    // create schedule
    GPUSchedule* dev_schedule;
    gpuErrchk( cudaMallocManaged((void**)&dev_schedule, sizeof(GPUSchedule)));
    dev_schedule->create_from_schedule(schedule);
    
    printf("schedule.prefix_num: %d\n", schedule.get_total_prefix_num());
    printf("shared memory for vertex set per block: %ld bytes\n", 
        (schedule.get_total_prefix_num() + 2) * WARPS_PER_BLOCK * sizeof(GPUVertexSet));


    tmpTime.print("Prepare time cost");
    tmpTime.check();

    uint32_t buffer_size = VertexSet::max_intersection_size;

    uint32_t block_shmem_size = (schedule.get_total_prefix_num() + 2) * WARPS_PER_BLOCK * sizeof(GPUVertexSet) + (max_edge + 1) * WARPS_PER_BLOCK * sizeof(char); // max_edge + 1是指一个pattern最多这么多点，用于存储p_label

    dev_schedule->p_label_offset = block_shmem_size - (max_edge + 1) * WARPS_PER_BLOCK * sizeof(char);
    dev_schedule->max_edge = max_edge;
    // 注意：此处没有错误，buffer_size代指每个顶点集所需的int数目，无需再乘sizeof(uint32_t)，但是否考虑对齐？
    //因为目前用了managed开内存，所以第一次运行kernel会有一定额外开销，考虑运行两次，第一次作为warmup
    
    int max_active_blocks_per_sm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks_per_sm, gpu_pattern_matching, THREADS_PER_BLOCK, block_shmem_size);
    printf("max number of active warps per SM: %d\n", max_active_blocks_per_sm * WARPS_PER_BLOCK);
    

    gpu_pattern_matching<<<num_blocks, THREADS_PER_BLOCK, block_shmem_size>>>
        (job_start, job_end, g->v_cnt, buffer_size, dev_edge, dev_labeled_vertex, dev_v_label, dev_tmp, dev_schedule, dev_all_p_label, fsm_set, dev_automorphisms, dev_is_frequent, dev_label_start_idx, automorphisms.size(), min_support, pattern_is_frequent_index, g->l_cnt);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk( cudaMemcpyFromSymbol(&sum, dev_sum, sizeof(sum)) );


    // 尝试释放一些内存
    gpuErrchk(cudaFree(dev_schedule->father_prefix_id));
    gpuErrchk(cudaFree(dev_schedule->last));
    gpuErrchk(cudaFree(dev_schedule->next));
    gpuErrchk(cudaFree(dev_schedule->loop_set_prefix_id));
    gpuErrchk(cudaFree(dev_schedule->prefix_target));

    gpuErrchk(cudaFree(dev_schedule));

    return sum;
}

void fsm_init(const LabeledGraph* g, int max_edge, int min_support) {
    std::vector<Pattern> patterns;
    Schedule_IEP* schedules;
    int schedules_num;
    int* mapping_start_idx;
    int* mappings;
    unsigned int* pattern_is_frequent_index; //每个unlabeled pattern对应的所有labeled pattern在is_frequent中的起始位置
    unsigned int* is_frequent; //bit vector
    unsigned int * tmp_is_frequent; // tmp for gpu to store
    g->get_fsm_necessary_info(patterns, max_edge, schedules, schedules_num, mapping_start_idx, mappings, pattern_is_frequent_index, is_frequent);
    long long fsm_cnt = 0;

    //特殊处理一个点的pattern
    for (int i = 0; i < g->l_cnt; ++i)
        if (g->label_frequency[i] >= min_support) {
            ++fsm_cnt;
            is_frequent[i >> 5] |= (unsigned int) (1 << (i % 32));
        }
    if (max_edge != 0)
        fsm_cnt = 0;

    size_t size_tmp_is_frequent = ((pattern_is_frequent_index[schedules_num] + 31) / 32) * sizeof(uint32_t);
    tmp_is_frequent = new unsigned int [size_tmp_is_frequent];

    int mapping_start_idx_pos = 1;

    size_t max_labeled_patterns = 1;
    for (int i = 0; i < max_edge + 1; ++i) //边数最大max_edge，点数最大max_edge + 1
        max_labeled_patterns *= (size_t) g->l_cnt;
    printf("max_labeled_patterns:%d\n", max_labeled_patterns);
    char* all_p_label = new char[max_labeled_patterns * (max_edge + 1) * 100];
    char* tmp_p_label = new char[(max_edge + 1) * 100];

    // 无关schedule的一些gpu初始化
    size_t size_edge = g->e_cnt * sizeof(uint32_t);
    size_t size_labeled_vertex = (g->v_cnt * g->l_cnt + 1) * sizeof(uint32_t);
    size_t size_v_label = g->v_cnt * sizeof(int);
    int max_total_prefix_num = 0;
    for (int i = 0; i < schedules_num; ++i)
    {
        schedules[i].update_loop_invariant_for_fsm();
        if (schedules[i].get_total_prefix_num() > max_total_prefix_num)
            max_total_prefix_num = schedules[i].get_total_prefix_num();
    }
    size_t size_tmp = VertexSet::max_intersection_size * sizeof(uint32_t) * num_total_warps * (max_total_prefix_num + 2); //prefix + subtraction + tmp
    size_t size_pattern_is_frequent_index = (schedules_num + 1) * sizeof(uint32_t);
    size_t size_is_frequent = ((pattern_is_frequent_index[schedules_num] + 31) / 32) * sizeof(uint32_t);
    size_t size_all_p_label = max_labeled_patterns * (max_edge + 1) * sizeof(char);
    size_t size_label_start_idx = (g->l_cnt + 1) * sizeof(uint32_t);

    uint32_t *dev_edge;
    //uint32_t *dev_edge_from;
    uint32_t *dev_labeled_vertex;
    int *dev_v_label;
    uint32_t *dev_tmp;
    uint32_t *dev_pattern_is_frequent_index;
    uint32_t *dev_is_frequent;
    char *dev_all_p_label;
    uint32_t *dev_label_start_idx;
    GPUBitVector* dev_fsm_set;

    // dev_alloc_and_copy((void **)&dev_edge, size_edge, g->edge);
    // dev_alloc_and_copy((void **)&dev_labeled_vertex, size_labeled_vertex, g->labeled_vertex);
    // dev_alloc_and_copy((void **)&dev_v_label, size_v_label, g->v_label);
    // dev_alloc_and_copy((void **)&dev_tmp, size_tmp);
    // dev_alloc_and_copy((void **)&dev_pattern_is_frequent_index, size_pattern_is_frequent_index, pattern_is_frequent_index);
    // dev_alloc_and_copy((void **)&dev_is_frequent, size_is_frequent, is_frequent);
    // dev_alloc_and_copy((void **)&dev_all_p_label, size_all_p_label);
    // dev_alloc_and_copy((void **)&dev_label_start_idx, size_label_start_idx, g->label_start_idx);

    gpuErrchk( cudaMalloc((void**)&dev_edge, size_edge));
    gpuErrchk( cudaMalloc((void**)&dev_labeled_vertex, size_labeled_vertex));
    gpuErrchk( cudaMalloc((void**)&dev_v_label, size_v_label));
    gpuErrchk( cudaMalloc((void**)&dev_tmp, size_tmp));
    gpuErrchk( cudaMalloc((void**)&dev_pattern_is_frequent_index, size_pattern_is_frequent_index));
    gpuErrchk( cudaMalloc((void**)&dev_is_frequent, size_is_frequent));
    gpuErrchk( cudaMalloc((void**)&dev_all_p_label, size_all_p_label));
    gpuErrchk( cudaMalloc((void**)&dev_label_start_idx, size_label_start_idx));

    gpuErrchk( cudaMemcpy(dev_edge, g->edge, size_edge, cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy(dev_labeled_vertex, g->labeled_vertex, size_labeled_vertex, cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy(dev_v_label, g->v_label, size_v_label, cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy(dev_pattern_is_frequent_index, pattern_is_frequent_index, size_pattern_is_frequent_index, cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy(dev_is_frequent, is_frequent, size_is_frequent, cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy(dev_label_start_idx, g->label_start_idx, size_label_start_idx, cudaMemcpyHostToDevice));

    //TODO: 之后考虑把fsm_set的成员变量放在shared memory，只把data内的数据放在global memory，就像vertex set一样
    gpuErrchk( cudaMallocManaged((void**)&dev_fsm_set, sizeof(GPUBitVector) * num_total_warps * (max_edge + 1))); //每个点一个fsm_set，一个pattern最多max_edge+1个点，每个warp负责一个不同的labeled pattern
    for (int i = 0; i < num_total_warps * (max_edge + 1); ++i)
        dev_fsm_set[i].construct(g->v_cnt);

    timeval start, end, total_time;
    gettimeofday(&start, NULL);

    printf("schedule num: %d\n", schedules_num);


    for (int i = 1; i < schedules_num; ++i) {
        std::vector<std::vector<int> > automorphisms;
        automorphisms.clear();
        schedules[i].GraphZero_get_automorphisms(automorphisms);
        //schedules[i].update_loop_invariant_for_fsm();
        size_t all_p_label_idx = 0;
        g->traverse_all_labeled_patterns(schedules, all_p_label, tmp_p_label, mapping_start_idx, mappings, pattern_is_frequent_index, is_frequent, i, 0, mapping_start_idx_pos, all_p_label_idx);
        printf("all_p_label_idx: %u\n", all_p_label_idx);
        gpuErrchk( cudaMemcpy(dev_all_p_label, all_p_label, all_p_label_idx * sizeof(char), cudaMemcpyHostToDevice));
        int job_num = all_p_label_idx / schedules[i].get_size();
        
        unsigned int cpu_jobs = std::min(job_num, std::max(int(job_num * cpu_proportion), 10));
        // cpu_jobs = 0.2;
        omp_set_nested(1);
        #pragma omp parallel num_threads(2)
        {
            int thread_id = omp_get_thread_num();
            if(thread_id == 0) {
                gpuErrchk( cudaMemcpyToSymbol(dev_cur_labeled_pattern, &cpu_jobs, sizeof(cpu_jobs)));
                int64_t ans = pattern_matching_init(g, schedules[i], automorphisms, pattern_is_frequent_index[i], dev_is_frequent, dev_edge, dev_labeled_vertex, dev_v_label, dev_tmp, max_edge, cpu_jobs, job_num, dev_all_p_label, dev_fsm_set, dev_label_start_idx, min_support);
                #pragma omp critical
                {
                    fsm_cnt += ans;
                }
            }
            else {
                int64_t ans = g->fsm_pattern_matching(0, cpu_jobs, schedules[i], all_p_label, automorphisms, is_frequent, pattern_is_frequent_index[i], max_edge, min_support, 16);
                #pragma omp critical
                {
                    fsm_cnt += ans;
                }
            }
        }
        printf("temp fsm_cnt: %lld, ", fsm_cnt);
        mapping_start_idx_pos += schedules[i].get_size();
        if (get_pattern_edge_num(patterns[i]) != max_edge) //为了使得边数小于max_edge的pattern不被统计。正确性依赖于pattern按照边数排序
            fsm_cnt = 0;
        assert(pattern_is_frequent_index[i] % 32 == 0);
        assert(pattern_is_frequent_index[i + 1] % 32 == 0);
        int is_frequent_index = pattern_is_frequent_index[i] / 32;
        size_t is_frequent_size = (pattern_is_frequent_index[i + 1] - pattern_is_frequent_index[i]) / 32 * sizeof(uint32_t);
        gpuErrchk( cudaMemcpy(&tmp_is_frequent[is_frequent_index], &dev_is_frequent[is_frequent_index], is_frequent_size, cudaMemcpyDeviceToHost));
        for(int i = 0; i < is_frequent_size; i++) {
            is_frequent[is_frequent_index + i] |= tmp_is_frequent[is_frequent_index + i];
        }
        printf("fsm_cnt: %ld\n",fsm_cnt);

        // 时间相关
        gettimeofday(&end, NULL);
        timersub(&end, &start, &total_time);
        printf("time = %ld s %06ld us.\n", total_time.tv_sec, total_time.tv_usec);
    }

    gpuErrchk(cudaFree(dev_edge));
    //gpuErrchk(cudaFree(dev_edge_from));
    gpuErrchk(cudaFree(dev_labeled_vertex));
    gpuErrchk(cudaFree(dev_v_label));
    gpuErrchk(cudaFree(dev_tmp));
    gpuErrchk(cudaFree(dev_pattern_is_frequent_index));
    gpuErrchk(cudaFree(dev_is_frequent));
    gpuErrchk(cudaFree(dev_all_p_label));
    gpuErrchk(cudaFree(dev_label_start_idx));
    for (int i = 0; i < num_total_warps * (max_edge + 1); ++i)
        dev_fsm_set[i].destroy();
    gpuErrchk(cudaFree(dev_fsm_set));


    printf("fsm cnt = %lld\n", fsm_cnt);

    free(schedules);
    delete[] mapping_start_idx;
    delete[] mappings;
    delete[] pattern_is_frequent_index;
    delete[] is_frequent;
    delete[] all_p_label;
    delete[] tmp_p_label;
}

int main(int argc,char *argv[]) {
    printf("file_name: %s\n", argv[0]);
    print_parameter();

    LabeledGraph *g;
    DataLoader D;

    const std::string path = argv[1];
    const int max_edge = atoi(argv[2]);
    const int min_support = atoi(argv[3]);

    DataType my_type;
    
    GetDataType(my_type, "Patents");

    if(my_type == DataType::Invalid) {
        printf("Dataset not found!\n");
        return 0;
    }

    g = new LabeledGraph();
    assert(D.load_labeled_data(g,my_type,path.c_str())==true);
    fsm_init(g, max_edge, min_support);

    return 0;
}
