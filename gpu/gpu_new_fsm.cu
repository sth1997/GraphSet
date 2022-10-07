// 第一种尝试，每个 labeled pattern 用一个 kernel
#define THRUST_IGNORE_CUB_VERSION_CHECK
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
#include <cub/cub.cuh>

#include <timeinterval.h>
#include "utils.cuh"
#include "gpu_schedule.cuh"
#include "gpu_fsm_vertex_set.cuh"
#include "gpu_bitvector.cuh"
#include "gpu_const.cuh"

__device__ unsigned long long dev_support = 0;



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
    const int wid = threadIdx.x / THREADS_PER_WARP;
    const int global_wid = blockIdx.x * WARPS_PER_BLOCK + wid; // global warp id   
    const int lid = threadIdx.x % THREADS_PER_WARP;
    int loop_set_prefix_id = schedule->get_loop_set_prefix_id(depth);
    int loop_size = vertex_set[loop_set_prefix_id].get_size();
    if (loop_size <= 0) //这个判断可能可以删了
        return false;
    uint32_t* loop_data_ptr = vertex_set[loop_set_prefix_id].get_data_ptr();

    bool local_match = false;
    __shared__ bool block_match[WARPS_PER_BLOCK];


    if (depth == schedule->get_size() - 1) {
        // warp 的线程一起做 insert
        for (int vertex_block = 0; vertex_block < loop_size; vertex_block += THREADS_PER_WARP)
        {
            if(vertex_block + lid >= loop_size) break;
            int vertex = loop_data_ptr[vertex_block + lid];
            if (subtraction_set.has_data_size(vertex, depth))
                continue;
            local_match = true;
            // for(int i = 0; i < WARPS_PER_BLOCK; i++) if(wid == i) {
                fsm_set[depth].insert(vertex);
            // }
            __threadfence_block();
        }
        __syncwarp();
        block_match[wid] = __any_sync(__activemask(), local_match);
        return block_match[wid]; 
    }

    for (int i = 0; i < loop_size; ++i)
    {
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
        if (lid == 0)
            subtraction_set.put(v, depth);
        __threadfence_block();

        if (GPU_pattern_matching_func<depth + 1>(schedule, vertex_set, subtraction_set, edge, labeled_vertex, p_label, fsm_set, l_cnt)) {
            local_match = true;
            if (threadIdx.x % THREADS_PER_WARP == 0) {
                fsm_set[depth].insert(v);
                __threadfence_block();
            }
        }
        // if (lid == 0)
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


__global__ void gpu_single_pattern_matching(uint32_t job_id, uint32_t v_cnt, uint32_t buffer_size, uint32_t *edge, uint32_t* labeled_vertex, int* v_label, uint32_t* tmp, const GPUSchedule* schedule, char* all_p_label, GPUBitVector* global_fsm_set, unsigned int* label_start_idx, long long min_support, int l_cnt, bool* break_indicater){
    extern __shared__ GPUVertexSet block_vertex_set[];

    *break_indicater = false;
    dev_support = 0;

    int wid = threadIdx.x / THREADS_PER_WARP; // warp id within the block
    int lid = threadIdx.x % THREADS_PER_WARP; // lane id
    int global_wid = blockIdx.x * WARPS_PER_BLOCK + wid; // global warp id   
    char* p_label = ((char*) (block_vertex_set)) + schedule->p_label_offset + (schedule->max_edge + 1) * wid; 
    int num_prefixes = schedule->get_total_prefix_num();
    int num_vertex_sets_per_warp = num_prefixes + 2;
    GPUVertexSet *vertex_set = block_vertex_set + wid * num_vertex_sets_per_warp;
    GPUBitVector *fsm_set = global_fsm_set + global_wid * schedule->get_size();
    // 这种 label 下开始的 vertex 和结束的 vertex 编号

    // FSM 使用的 vector，记录 support
    GPUVertexSet& subtraction_set = vertex_set[num_prefixes];
    
    if (lid == 0) {
        // set vertex_set's memory
        uint32_t offset = buffer_size * global_wid * num_vertex_sets_per_warp;
        for (int i = 0; i < num_vertex_sets_per_warp; ++i)
        {
            vertex_set[i].set_data_ptr(tmp + offset); // 注意这是个指针+整数运算，自带*4
            offset += buffer_size;
        }
        subtraction_set.init();
        size_t job_start_idx = job_id * schedule->get_size();
        for (int j = 0; j < schedule->get_size(); ++j)
            p_label[j] = all_p_label[job_start_idx + j];
        // assert(global_wid >= schedule->get_size())
    }

    if (lid < schedule->get_size())
        fsm_set[lid].clear();
    // if(global_wid == 0 && lid == 0) {
    //     printf("pattern's label: ");
    //     for(int i = 0; i < schedule->size; i++){
    //         printf("%d ",p_label[i]);
    //     }
    //     printf("\n");
    // }

    __threadfence_block();

    int start_v = label_start_idx[p_label[0]], end_v = label_start_idx[p_label[0] + 1];
    for(int vertex_block = start_v; vertex_block < end_v; vertex_block += num_total_warps) {
        // if(global_wid == 0 && lid == 0)
        //     printf("vertex_block: %d\n", vertex_block);
        int vertex_id = vertex_block + global_wid;
        if (vertex_id >= end_v) break;

        if(*break_indicater) break;

        bool is_zero = false;
        for (int prefix_id = schedule->get_last(0); prefix_id != -1; prefix_id = schedule->get_next(prefix_id)) {
            unsigned int l, r;
            int target = schedule->get_prefix_target(prefix_id);
            get_labeled_edge_index(vertex_id, p_label[target], l, r);
            vertex_set[prefix_id].build_vertex_set(schedule, vertex_set, &edge[l], (int)r - l, prefix_id);
            if (vertex_set[prefix_id].get_size() == 0) {
                is_zero = true;
                break;
            }
        }
        if (is_zero)
            continue;
        if (lid == 0)
            subtraction_set.put(vertex_id, 0);
        
        __threadfence_block();


        if(GPU_pattern_matching_func<1>(schedule, vertex_set, subtraction_set, edge, labeled_vertex, p_label, fsm_set, l_cnt)) {
            if(lid == 0){
                // for(int i = 0; i < num_total_warps; i++) if(i == global_wid) {
                    fsm_set[0].insert(vertex_id);
                // }
            }
        }

        // if (lid == 0)
        //     subtraction_set.pop_back();
        __threadfence_block();

        // // try to cut down unneceseary ones

        // long long support = v_cnt;
        // for (int i = 0; i < schedule->size; i++){
        //     long long count = fsm_set[i].get_non_zero_cnt();
        //     if(count < support) support = count;
        // }
        // if(lid == 0) {
        //     if(support > min_support) {
        //         *break_indicater = true;
        //     }
        // }
    }

    // cudaFree(temp_sum_counter);
}

class OrOperator{
    public:
    __device__ uint32_t operator() (const uint32_t &a, const uint32_t &b) const {
        return a | b;
    }
};

__global__ void reduce_fsm_set(GPUBitVector *global_fsm_set, uint32_t bit_vector_size, uint32_t schedule_size, int32_t *output) {
    int wid = threadIdx.x / THREADS_PER_WARP; // warp id within the block
    int lid = threadIdx.x % THREADS_PER_WARP; // lane id
    int global_wid = blockIdx.x * WARPS_PER_BLOCK + wid; // global warp id  
    // GPUBitVector *output = global_fsm_set + schedule_size * num_total_warps;
    
    // if(lid == 0 && global_wid < schedule_size) 
    //     // output[global_wid].clear();
    //     output[global_wid] = 0;
    
    __syncthreads();
    __threadfence();

    typedef cub::WarpReduce<uint32_t, THREADS_PER_WARP> WarpReduce;
    __shared__ typename WarpReduce::TempStorage temp_storage[WARPS_PER_BLOCK];
    // __shared__ uint32_t answer[THREADS_PER_BLOCK];

    // brute force version
    // for(int pos = 0; pos < schedule_size * bit_vector_size; pos++){
    //     int t = pos / bit_vector_size;
    //     int p = pos % bit_vector_size;    
    //     uint32_t tmp_result = 0;
    //     for(int i = 0; i < num_total_warps; i ++) {
    //         tmp_result |= (*(global_fsm_set + i * schedule_size + t)).get_data()[p];
    //         __syncwarp();
    //     }
    //     __threadfence();
    //     output[t].get_data()[p] = tmp_result;        
    // }

    for(int s_block = 0; s_block < schedule_size * bit_vector_size; s_block += num_total_warps) {
        int pos = s_block + global_wid;
        int t = pos / bit_vector_size;
        int p = pos % bit_vector_size;
        if(pos >= schedule_size * bit_vector_size) break;
        
        uint32_t tmp_result = 0;
        // for(int i = 0; i < num_total_warps; i ++) {
        //     tmp_result |= (*(global_fsm_set + i * schedule_size + t)).get_data()[p];
        // }
        // output[t].get_data()[p] = tmp_result;
        for(int i_block = 0; i_block < num_total_warps; i_block += WARPS_PER_BLOCK) {
            int i = i_block + lid;
            if(i >= num_total_warps) break;
            tmp_result |= (*(global_fsm_set + i * schedule_size + t)).get_data()[p];
        }
        __syncwarp();
        uint32_t agg = WarpReduce(temp_storage[wid]).Reduce(tmp_result, OrOperator());
        __syncwarp();
        if(lid == 0) {
            atomicAdd(&output[t], __popc(agg));
            // output[t].get_data()[p] = agg;
        }
        __threadfence_system();
    }
}



long long pattern_matching_init(const LabeledGraph *g, const Schedule_IEP& schedule, const std::vector<std::vector<int> >& automorphisms, unsigned int pattern_is_frequent_index, unsigned int* is_frequent, uint32_t* dev_edge, uint32_t* dev_labeled_vertex, int* dev_v_label, uint32_t* dev_tmp, int max_edge, int job_num, char* all_p_label, char* dev_all_p_label, GPUBitVector* dev_fsm_set, uint32_t* dev_label_start_idx, long long min_support) {

    // printf("enter pattern matching init\n");
    printf("total prefix %d\n", schedule.get_total_prefix_num());
    schedule.print_schedule();
    fflush(stdout);

    tmpTime.check(); 

    long long sum = 0; //sum是这个pattern的所有labeled pattern中频繁的个数

    //memcpy schedule
    GPUSchedule* dev_schedule;
    gpuErrchk( cudaMallocManaged((void**)&dev_schedule, sizeof(GPUSchedule)));
    //dev_schedule->transform_in_exclusion_optimize_group_val(schedule);
    int schedule_size = schedule.get_size();
    int max_prefix_num = schedule_size * (schedule_size - 1) / 2;

    gpuErrchk( cudaMallocManaged((void**)&dev_schedule->father_prefix_id, sizeof(int) * max_prefix_num));
    gpuErrchk( cudaMemcpy(dev_schedule->father_prefix_id, schedule.get_father_prefix_id_ptr(), sizeof(int) * max_prefix_num, cudaMemcpyHostToDevice));

    gpuErrchk( cudaMallocManaged((void**)&dev_schedule->last, sizeof(int) * schedule_size));
    gpuErrchk( cudaMemcpy(dev_schedule->last, schedule.get_last_ptr(), sizeof(int) * schedule_size, cudaMemcpyHostToDevice));

    gpuErrchk( cudaMallocManaged((void**)&dev_schedule->next, sizeof(int) * max_prefix_num));
    gpuErrchk( cudaMemcpy(dev_schedule->next, schedule.get_next_ptr(), sizeof(int) * max_prefix_num, cudaMemcpyHostToDevice));

    gpuErrchk( cudaMallocManaged((void**)&dev_schedule->loop_set_prefix_id, sizeof(int) * schedule_size));
    gpuErrchk( cudaMemcpy(dev_schedule->loop_set_prefix_id, schedule.get_loop_set_prefix_id_ptr(), sizeof(int) * schedule_size, cudaMemcpyHostToDevice));

    gpuErrchk( cudaMallocManaged((void**)&dev_schedule->prefix_target, sizeof(int) * max_prefix_num));
    gpuErrchk( cudaMemcpy(dev_schedule->prefix_target, schedule.get_prefix_target_ptr(), sizeof(int) * max_prefix_num, cudaMemcpyHostToDevice));

    dev_schedule->size = schedule.get_size();
    dev_schedule->total_prefix_num = schedule.get_total_prefix_num();
    
    printf("schedule.prefix_num: %d\n", schedule.get_total_prefix_num());
    printf("shared memory for vertex set per block: %ld bytes\n", 
        (schedule.get_total_prefix_num() + 2) * WARPS_PER_BLOCK * sizeof(GPUVertexSet));


    tmpTime.print("Prepare time cost");
    tmpTime.check();

    uint32_t buffer_size = VertexSet::max_intersection_size;
    //uint32_t block_shmem_size = (schedule.get_total_prefix_num() + 2) * WARPS_PER_BLOCK * sizeof(GPUVertexSet) + in_exclusion_optimize_vertex_id_size * WARPS_PER_BLOCK * sizeof(int);
    uint32_t block_shmem_size = (schedule.get_total_prefix_num() + 2) * WARPS_PER_BLOCK * sizeof(GPUVertexSet) + (max_edge + 1) * WARPS_PER_BLOCK * sizeof(char); // max_edge + 1是指一个pattern最多这么多点，用于存储p_label
    //dev_schedule->ans_array_offset = block_shmem_size - in_exclusion_optimize_vertex_id_size * WARPS_PER_BLOCK * sizeof(int);
    dev_schedule->p_label_offset = block_shmem_size - (max_edge + 1) * WARPS_PER_BLOCK * sizeof(char);
    dev_schedule->max_edge = max_edge;
    // 注意：此处没有错误，buffer_size代指每个顶点集所需的int数目，无需再乘sizeof(uint32_t)，但是否考虑对齐？
    //因为目前用了managed开内存，所以第一次运行kernel会有一定额外开销，考虑运行两次，第一次作为warmup
    
    int max_active_blocks_per_sm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks_per_sm, gpu_single_pattern_matching, THREADS_PER_BLOCK, block_shmem_size);
    printf("max number of active warps per SM: %d\n", max_active_blocks_per_sm * WARPS_PER_BLOCK);
    fflush(stdout);

    printf("total_job_num: %d\n", job_num);

    for(int job_id = 0; job_id < job_num; job_id++){
        // if(job_id % 100 == 0)
        //     printf("job id: %d/%d\n",job_id, job_num);
        fflush(stdout);

        bool * break_indicater, result;

        int * dev_fsm_set_size;
        
        gpuErrchk(cudaMalloc((void**)&break_indicater, sizeof(bool)));
        gpuErrchk(cudaMallocManaged((void**)&dev_fsm_set_size, sizeof(int) * schedule_size));
        memset(dev_fsm_set_size, 0, sizeof(int) * schedule_size);

        // a *single* labeled pattern here
        gpu_single_pattern_matching<<<num_blocks, THREADS_PER_BLOCK, block_shmem_size>>>(job_id, g->v_cnt, buffer_size, dev_edge, dev_labeled_vertex, dev_v_label, dev_tmp, dev_schedule, dev_all_p_label, dev_fsm_set, dev_label_start_idx, min_support, g->l_cnt, break_indicater);

        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        gpuErrchk(cudaMemcpy(&result, break_indicater, sizeof(bool), cudaMemcpyDeviceToHost));

        long long support_answer = g->v_cnt;
        
        if(!result) {
            reduce_fsm_set<<<num_blocks, THREADS_PER_BLOCK>>>(dev_fsm_set, (g->v_cnt+31)/32, schedule_size, dev_fsm_set_size);

            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );
            for(int i = 0; i < schedule.get_size(); i++) {
                printf("fsm_set[%d]:%lld\n", i, dev_fsm_set_size[i]);
                if(dev_fsm_set_size[i] < support_answer) support_answer = dev_fsm_set_size[i];
            }
            printf("finish job %d, support answer:%lld\n", job_id, support_answer);
            fflush(stdout);
        }

        if (result || support_answer >= min_support) {
            sum += 1;
            char* p_label = all_p_label + job_id * schedule.get_size(); 
            for (int aut_id = 0; aut_id < automorphisms.size(); ++aut_id) {
                const std::vector<int> & aut = automorphisms[aut_id];
                unsigned int index = pattern_is_frequent_index;
                unsigned int pow = 1;
                for (int j = 0; j < schedule.get_size(); ++j) {
                    index += p_label[ aut[j] ] * pow;
                    pow *= (unsigned int) g->l_cnt;
                }
                is_frequent[index >> 5] |= (1 << (index % 32));
            }
        }
    }

    printf("job_id: %d/%d sum:%lld\n", job_num, job_num, sum);
    fflush(stdout);

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
    size_t size_all_p_label = max_labeled_patterns * (max_edge + 1) * sizeof(char);
    size_t size_label_start_idx = (g->l_cnt + 1) * sizeof(uint32_t);

    uint32_t *dev_edge;
    uint32_t *dev_labeled_vertex;
    int *dev_v_label;
    uint32_t *dev_tmp;
    char *dev_all_p_label;
    uint32_t *dev_label_start_idx;
    GPUBitVector* dev_fsm_set;

    gpuErrchk( cudaMalloc((void**)&dev_edge, size_edge));
    gpuErrchk( cudaMalloc((void**)&dev_labeled_vertex, size_labeled_vertex));
    gpuErrchk( cudaMalloc((void**)&dev_v_label, size_v_label));
    gpuErrchk( cudaMalloc((void**)&dev_tmp, size_tmp));
    gpuErrchk( cudaMalloc((void**)&dev_all_p_label, size_all_p_label));
    gpuErrchk( cudaMalloc((void**)&dev_label_start_idx, size_label_start_idx));

    gpuErrchk( cudaMemcpy(dev_edge, g->edge, size_edge, cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy(dev_labeled_vertex, g->labeled_vertex, size_labeled_vertex, cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy(dev_v_label, g->v_label, size_v_label, cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy(dev_label_start_idx, g->label_start_idx, size_label_start_idx, cudaMemcpyHostToDevice));

    gpuErrchk( cudaMallocManaged((void**)&dev_fsm_set, sizeof(GPUBitVector) * (num_total_warps + 1) * (max_edge + 1)));
    for (int i = 0; i < (num_total_warps + 1) * (max_edge + 1); ++i)
        dev_fsm_set[i].construct(g->v_cnt);

    timeval start, end, total_time;
    gettimeofday(&start, NULL);

    printf("schedule num: %d\n", schedules_num);


    for (int i = 1; i < schedules_num; ++i) {
        std::vector<std::vector<int> > automorphisms;
        automorphisms.clear();
        schedules[i].GraphZero_get_automorphisms(automorphisms);
        size_t all_p_label_idx = 0;
        g->traverse_all_labeled_patterns(schedules, all_p_label, tmp_p_label, mapping_start_idx, mappings, pattern_is_frequent_index, is_frequent, i, 0, mapping_start_idx_pos, all_p_label_idx);
        printf("all_p_label_idx: %u\n", all_p_label_idx);
        gpuErrchk( cudaMemcpy(dev_all_p_label, all_p_label, all_p_label_idx * sizeof(char), cudaMemcpyHostToDevice));
        int job_num = all_p_label_idx / schedules[i].get_size();

        fflush(stdout);

        fsm_cnt += pattern_matching_init(g, schedules[i], automorphisms, pattern_is_frequent_index[i], is_frequent, dev_edge, dev_labeled_vertex, dev_v_label, dev_tmp, max_edge, job_num, all_p_label, dev_all_p_label, dev_fsm_set, dev_label_start_idx, min_support);
        mapping_start_idx_pos += schedules[i].get_size();

        printf("temp fsm_cnt: %lld\n", fsm_cnt);

        if (get_pattern_edge_num(patterns[i]) != max_edge) //为了使得边数小于max_edge的pattern不被统计。正确性依赖于pattern按照边数排序
            fsm_cnt = 0;

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
    gpuErrchk(cudaFree(dev_all_p_label));
    gpuErrchk(cudaFree(dev_label_start_idx));
    for (int i = 0; i < (max_edge + 1); ++i)
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
    LabeledGraph *g;
    DataLoader D;

    // const std::string type = argv[1];
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
