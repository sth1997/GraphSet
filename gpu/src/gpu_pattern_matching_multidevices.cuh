#pragma once
#include "../component/gpu_device_context.cuh"
#include "../component/gpu_schedule.cuh"
#include "../component/gpu_vertex_set.cuh"
#include "../component/utils.cuh"

#include "../function/pattern_matching.cuh"

struct TaskItem {
    volatile uint32_t *task_fetched;
    volatile unsigned long long *new_task_start, *new_task_end;
    unsigned long long *task_start, *task_end;
    void init() {
        gpuErrchk( cudaMallocManaged((void**)&task_fetched, sizeof(uint32_t)));
        gpuErrchk( cudaMallocManaged((void**)&new_task_start, sizeof(unsigned long long)));
        gpuErrchk( cudaMallocManaged((void**)&new_task_end, sizeof(unsigned long long)));
        gpuErrchk( cudaMallocManaged((void**)&task_start, sizeof(unsigned long long)));
        gpuErrchk( cudaMallocManaged((void**)&task_end, sizeof(unsigned long long)));
        *task_fetched = 1;
        *new_task_start = *new_task_end = -1;
        *task_start = *task_end = -1;
    }
    void destroy() {
        gpuErrchk( cudaFree((void *)task_fetched) );
        gpuErrchk( cudaFree((void *)new_task_start) );
        gpuErrchk( cudaFree((void *)new_task_end) );
        gpuErrchk( cudaFree((void *)task_start) );
        gpuErrchk( cudaFree((void *)task_end) );
    }
};


template <uint32_t val> 
__device__ void acquire_lock_device(uint32_t *lock) {
    while (true) {
        // whether that can make some "deadlock"? 
        // no other instructions can insert into the loops?
        uint32_t result = atomicCAS(lock, 0u, val);
        if(result == val) break;
    }
}

template <uint32_t val>
__device__ void release_lock_device(uint32_t * lock) {
    atomicOr(lock, 0);
}

/**
 * @note `buffer_size`实际上是每个节点的最大邻居数量，而非所用空间大小
 *
 */
__global__ void gpu_pattern_matching_multidevices(TaskItem* task, uint32_t buffer_size, PatternMatchingDeviceContext *context) {
    __shared__ e_index_t block_edge_idx[WARPS_PER_BLOCK];
    //之后考虑把tmp buffer都放到shared里来（如果放得下）
    extern __shared__ GPUVertexSet block_vertex_set[];

    GPUSchedule *schedule = context->dev_schedule;
    uint32_t *tmp = context->dev_tmp;
    uint32_t *edge = (uint32_t *)context->dev_edge;
    e_index_t *vertex = context->dev_vertex;
    uint32_t *edge_from = (uint32_t *)context->dev_edge_from;

    int num_prefixes = schedule->get_total_prefix_num();
    int num_vertex_sets_per_warp = num_prefixes + 2;

    int wid = threadIdx.x / THREADS_PER_WARP;            // warp id within the block
    int lid = threadIdx.x % THREADS_PER_WARP;            // lane id
    int global_wid = blockIdx.x * WARPS_PER_BLOCK + wid; // global warp id
    e_index_t &edge_idx = block_edge_idx[wid];
    GPUVertexSet *vertex_set = block_vertex_set + wid * num_vertex_sets_per_warp;
    if(global_wid == 0 && lid == 0) {
        printf("kernel launched\n\n");
    }

    if (lid == 0) {
        edge_idx = 0;
        uint32_t offset = buffer_size * global_wid * num_vertex_sets_per_warp;
        for (int i = 0; i < num_vertex_sets_per_warp; ++i) {
            vertex_set[i].set_data_ptr(tmp + offset); // 注意这是个指针+整数运算，自带*4
            offset += buffer_size;
        }
    }
    GPUVertexSet &subtraction_set = vertex_set[num_prefixes];
    GPUVertexSet &tmp_set = vertex_set[num_prefixes + 1];

    __threadfence_block(); //之后考虑把所有的syncthreads都改成syncwarp

    uint32_t v0, v1;
    e_index_t l, r;

    unsigned long long sum = 0;

    uint64_t last_task_end = -1;    

    while(true){
        __syncwarp();        
        // if(lid == 0 && (int)*(task->task_fetched) == 0) {
        //     printf("fetched: %d\n", (int)*(task->task_fetched));
        // }
        if(*task->new_task_end == 0) {
            if(global_wid == 0 && lid == 0) {
                printf("break\n");
            }   
            break;
        }
        if((int)(*(task->task_fetched)) == 1) continue;
        if(global_wid == 0 && lid == 0) {
            printf("task: %llu %llu\n", *task->new_task_start, *task->new_task_end);
        }
        // if(*task->new_task_end == last_task_end) continue;
        *task->task_start = *(task->new_task_start);
        *task->task_end = *(task->new_task_end);
        unsigned long long &dev_cur_edge = *(task->task_start);
        unsigned long long &edge_num = *(task->task_end);
        last_task_end = edge_num;
        *(task->task_fetched) = 1;
        

        // if(global_wid == 0 && lid == 0) {
        //     printf("task: %d %d\n", (int)dev_cur_edge, (int)edge_num);
        // }

        __threadfence_system();

        while (true) {
            if (lid == 0) {
                edge_idx = atomicAdd(&dev_cur_edge, 1);
                e_index_t i = edge_idx;
                if (i < edge_num) {
                    subtraction_set.init();
                    subtraction_set.push_back(edge_from[i]);
                    subtraction_set.push_back(edge[i]);
                }
            }

            __threadfence_block();

            e_index_t i = edge_idx;
            if (i >= edge_num)
                break;

            // for edge in E
            v0 = edge_from[i];
            v1 = edge[i];

            //目前只考虑pattern size>2的情况
            // start v1, depth = 1
            if (schedule->get_restrict_last(1) != -1 && v0 <= v1)
                continue;

            bool is_zero = false;
            get_edge_index(v0, l, r);
            for (int prefix_id = schedule->get_last(0); prefix_id != -1; prefix_id = schedule->get_next(prefix_id))
                vertex_set[prefix_id].build_vertex_set(schedule, vertex_set, &edge[l], r - l, prefix_id);

            get_edge_index(v1, l, r);
            for (int prefix_id = schedule->get_last(1); prefix_id != -1; prefix_id = schedule->get_next(prefix_id)) {
                vertex_set[prefix_id].build_vertex_set(schedule, vertex_set, &edge[l], r - l, prefix_id);
                if (vertex_set[prefix_id].get_size() == 0 && prefix_id < schedule->get_basic_prefix_num()) {
                    is_zero = true;
                    break;
                }
            }
            if (is_zero)
                continue;

            unsigned long long local_sum = 0; // local sum (corresponding to an edge index)
            GPU_pattern_matching_func<2>(schedule, vertex_set, subtraction_set, tmp_set, local_sum, edge, vertex);
            sum += local_sum;
        }
    }

    if (lid == 0) {
        atomicAdd(context->dev_sum, sum);
    }
}