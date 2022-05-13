/**
 * 这个版本里面没有细粒度计时。有计时的在gpu_graph_with_timer.cu里面。
 * 而且计时的方式与zms版本略有区别。
*/
#define THRUST_IGNORE_CUB_VERSION_CHECK
#include <graph.h>
#include <dataloader.h>
#include <vertex_set.h>
#include <common.h>
#include <schedule_IEP.h>
#include <cub/cub.cuh>

#include <cassert>
#include <cstring>
#include <cstdint>
#include <string>
#include <algorithm>
#include <unordered_set>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <sys/time.h>
#include <chrono>

#define likely(x)   __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
constexpr int THREADS_PER_BLOCK = 256;
constexpr int THREADS_PER_WARP = 32;
constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / THREADS_PER_WARP;

//#define PRINT_ANS_TO_FILE //用于scripts/small_graph_check.py

// 是否要用<chrono>中的内容进行替代？
class TimeInterval{
public:
    TimeInterval(){
        check();
    }

    void check(){
        gettimeofday(&tp, NULL);
    }

    void print(const char* title){
        struct timeval tp_end, tp_res;
        gettimeofday(&tp_end, NULL);
        timersub(&tp_end, &tp, &tp_res);
        printf("%s: %ld s %06ld us.\n", title, tp_res.tv_sec, tp_res.tv_usec);
    }
private:
    struct timeval tp;
};

TimeInterval allTime;
TimeInterval tmpTime;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
if (code != cudaSuccess) 
{
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
}
}

#define get_edge_index(v, l, r) do { \
    l = vertex[v]; \
    r = vertex[v + 1]; \
} while(0)

template <typename T>
__device__ inline void swap(T& a, T& b)
{
    T t(std::move(a));
    a = std::move(b);
    b = std::move(t);
}

struct GPUGroupDim2 {
    int* data;
    int size;
};

struct GPUGroupDim1 {
    GPUGroupDim2* data;
    int size;
};

struct GPUGroupDim0 {
    GPUGroupDim1* data;
    int size;
};

class GPUSchedule {
public:
    inline __device__ int get_total_prefix_num() const { return total_prefix_num;}
    inline __device__ int get_basic_prefix_num() const { return basic_prefix_num;}
    inline __device__ int get_father_prefix_id(int prefix_id) const { return father_prefix_id[prefix_id];}
    inline __device__ int get_loop_set_prefix_id(int loop) const { return loop_set_prefix_id[loop];}
    inline __device__ int get_size() const { return size;}
    inline __device__ int get_last(int i) const { return last[i];}
    inline __device__ int get_next(int i) const { return next[i];}
    inline __device__ int get_break_size(int i) const { return break_size[i];}
    inline __device__ int get_in_exclusion_optimize_num() const { return in_exclusion_optimize_num;}
    inline __device__ int get_total_restrict_num() const { return total_restrict_num;}
    inline __device__ int get_restrict_last(int i) const { return restrict_last[i];}
    inline __device__ int get_restrict_next(int i) const { return restrict_next[i];}
    inline __device__ int get_restrict_index(int i) const { return restrict_index[i];}
    //inline __device__ int get_k_val() const { return k_val;} // see below (the k_val's definition line) before using this function

    int* adj_mat;
    int* father_prefix_id;
    int* last;
    int* next;
    int* break_size;
    int* loop_set_prefix_id;
    int* restrict_last;
    int* restrict_next;
    int* restrict_index;
    bool* only_need_size;
    //int* in_exclusion_optimize_val;
    //GPUGroupDim0 in_exclusion_optimize_group;
    //int in_exclusion_optimize_val_size;
    int size;
    int total_prefix_num;
    int basic_prefix_num;
    int total_restrict_num;
    int in_exclusion_optimize_num;
    //int k_val;

    int in_exclusion_optimize_vertex_id_size;
    int* in_exclusion_optimize_vertex_id;
    bool* in_exclusion_optimize_vertex_flag;
    int* in_exclusion_optimize_vertex_coef;
    
    int in_exclusion_optimize_array_size;
    int* in_exclusion_optimize_coef;
    bool* in_exclusion_optimize_flag;
    int* in_exclusion_optimize_ans_pos;

    uint32_t ans_array_offset;
};

// __device__ void intersection1(uint32_t *tmp, uint32_t *lbases, uint32_t *rbases, uint32_t ln, uint32_t rn, uint32_t* p_tmp_size);
__device__ void intersection2(uint32_t *tmp, const uint32_t *lbases, const uint32_t *rbases, uint32_t ln, uint32_t rn, uint32_t* p_tmp_size);
static __device__ uint32_t do_intersection(uint32_t*, const uint32_t*, const uint32_t*, uint32_t, uint32_t);
class GPUVertexSet;
__device__ int unordered_subtraction_size(const GPUVertexSet& set0, const GPUVertexSet& set1, int size_after_restrict);




class GPUVertexSet
{
public:
    __device__ GPUVertexSet()
    {
        size = 0;
        data = NULL;
    }
    __device__ int get_size() const { return size;}
    __device__ uint32_t get_data(int i) const { return data[i];}
    __device__ void push_back(uint32_t val) { data[size++] = val;}
    __device__ void pop_back() { --size;}
    __device__ uint32_t get_last() const {return data[size - 1];}
    __device__ void set_data_ptr(uint32_t* ptr) { data = ptr;}
    __device__ uint32_t* get_data_ptr() const { return data;}
    __device__ bool has_data (uint32_t val) const // 注意：这里不用二分，调用它的是较小的无序集合
    {
        for (int i = 0; i < size; ++i)
            if (data[i] == val)
                return true;
        return false;
    }
    __device__ void init() { size = 0; }
    __device__ void init(uint32_t input_size, uint32_t* input_data)
    {
        size = input_size;
        data = input_data; //之后如果把所有prefix放到shared memory，由于input data在global memory上（因为是原图的边集），所以改成memcpy
    }
    __device__ void copy_from(const GPUVertexSet& other)//考虑改为并行
    {
        // 这个版本可能会有bank conflict
        uint32_t input_size = other.get_size(), *input_data = other.get_data_ptr();
        size = input_size;
        int lid = threadIdx.x % THREADS_PER_WARP; // lane id
        int size_per_thread = (input_size + THREADS_PER_WARP - 1) / THREADS_PER_WARP;
        int start = size_per_thread * lid;
        int end = min(start + size_per_thread, input_size);
        for (int i = start; i < end; ++i)
            data[i] = input_data[i];
        __threadfence_block();
    }
    __device__ void build_vertex_set(const GPUSchedule* schedule, const GPUVertexSet* vertex_set, uint32_t* input_data, uint32_t input_size, int prefix_id)
    {
        int father_id = schedule->get_father_prefix_id(prefix_id);
        if (father_id == -1)
        {
            if (threadIdx.x % THREADS_PER_WARP == 0)
                init(input_size, input_data);
            __threadfence_block();
        }
        else
        {
            // bool only_need_size = schedule->only_need_size[prefix_id];
            // if(only_need_size) {
            //     if (threadIdx.x % THREADS_PER_WARP == 0)
            //         init(input_size, input_data);
            //     __threadfence_block();
            //     if(input_size > vertex_set[father_id].get_size())
            //         this->size -= unordered_subtraction_size(*this, vertex_set[father_id], -1);
            //     else
            //         this->size = vertex_set[father_id].get_size() - unordered_subtraction_size(vertex_set[father_id], *this, -1);
            // }
            // else {
                intersection2(this->data, vertex_set[father_id].get_data_ptr(), input_data, vertex_set[father_id].get_size(), input_size, &this->size);
            // }
        }
    }


    __device__ void intersection_with(const GPUVertexSet& other)
    {
        uint32_t ret = do_intersection(data, data, other.get_data_ptr(), size, other.get_size());
        if (threadIdx.x % THREADS_PER_WARP == 0)
            size = ret;
        __threadfence_block();
    }

private:
    uint32_t size;
    uint32_t* data;
};

__device__ unsigned long long dev_sum = 0;
__device__ unsigned int dev_cur_edge = 0;

/**
 * search-based intersection
 * 
 * returns the size of the intersection set
 * 
 * @note：a和b并不是地位相等的。如果要进行in-place操作，请把输入放在a而不是b。
 * @todo：shared memory缓存优化
 */
__device__ uint32_t do_intersection(uint32_t* out, const uint32_t* a, const uint32_t* b, uint32_t na, uint32_t nb)
{
    // __shared__ uint32_t block_out_offset[THREADS_PER_BLOCK];
    __shared__ uint32_t block_out_size[WARPS_PER_BLOCK];
    typedef cub::WarpScan<uint8_t> WarpScan;
    __shared__ typename WarpScan::TempStorage temp_storage[WARPS_PER_BLOCK];

    int wid = threadIdx.x / THREADS_PER_WARP; // warp id
    int lid = threadIdx.x % THREADS_PER_WARP; // lane id
    // uint32_t *out_offset = block_out_offset + wid * THREADS_PER_WARP;
    uint32_t &out_size = block_out_size[wid];

    if (lid == 0)
        out_size = 0;

    for(int num_done = 0; num_done < na; num_done += THREADS_PER_WARP) {
        bool found = 0;
        uint32_t u = 0;
        if (num_done + lid < na) {
            u = a[num_done + lid]; // u: an element in set a
            int mid, l = 0, r = int(nb) - 1;
            while (l <= r) {
                mid = (l + r) >> 1;
                if (b[mid] < u) {
                    l = mid + 1;
                } else if (b[mid] > u) {
                    r = mid - 1;
                } else {
                    found = 1;
                    break;
                }
            }
        }

        uint8_t thread_data = found;

        WarpScan(temp_storage[wid]).InclusiveSum(thread_data, thread_data);
        
        if (found) {
            out[out_size + thread_data - 1] = u;
        }

        // __syncwarp();

        if (lid == THREADS_PER_WARP - 1)
            out_size += thread_data;
    }

    __threadfence_block();
    return out_size;
}




__device__ void intersection2(uint32_t *tmp, const uint32_t *lbases, const uint32_t *rbases, uint32_t ln, uint32_t rn, uint32_t* p_tmp_size)
{
    // make sure ln <= rn
    if (ln > rn) {
        swap(ln, rn);
        swap(lbases, rbases);
    }
    /**
     * @todo 考虑ln < rn <= 32时，每个线程在lbases里面找rbases的一个元素可能会更快
     */

    uint32_t intersection_size = do_intersection(tmp, lbases, rbases, ln, rn);

    if (threadIdx.x % THREADS_PER_WARP == 0)
        *p_tmp_size = intersection_size;
    __threadfence_block();
}


__device__ int lower_bound(const uint32_t* loop_data_ptr, int loop_size, int min_vertex)
{
    int l = 0, r = loop_size - 1;
    while (l <= r)
    {
        int mid = r - ((r - l) >> 1);
        if (loop_data_ptr[mid] < min_vertex)
            l = mid + 1;
        else
            r = mid - 1;
    }
    return l;
}

constexpr int MAX_DEPTH = 5; // 非递归pattern matching支持的最大深度

template <int depth>
__device__ void GPU_pattern_matching_func(const GPUSchedule* schedule, GPUVertexSet* vertex_set, unsigned long long& local_ans, uint32_t *edge, e_index_t *vertex)
{

    if (depth == schedule->get_size() - schedule->get_in_exclusion_optimize_num()) {
        assert(false);
        // GPU_pattern_matching_final_in_exclusion(schedule, vertex_set, subtraction_set, tmp_set, local_ans,  edge, vertex);
        return;    
    }

    int loop_set_prefix_id = schedule->get_loop_set_prefix_id(depth);
    int loop_size = vertex_set[loop_set_prefix_id].get_size();

    uint32_t* loop_data_ptr = vertex_set[loop_set_prefix_id].get_data_ptr();
    if (depth == schedule->get_size() - 1 && schedule->get_in_exclusion_optimize_num() == 0) {
        local_ans += loop_size;
        return;
    }

    for (int i = 0; i < loop_size; ++i)
    {
        uint32_t v = loop_data_ptr[i];
        long long l, r;
        get_edge_index(v, l, r);
        bool is_zero = false;
        for (int prefix_id = schedule->get_last(depth); prefix_id != -1; prefix_id = schedule->get_next(prefix_id))
        {
            vertex_set[prefix_id].build_vertex_set(schedule, vertex_set, &edge[l], (int)(r - l), prefix_id);
            if (vertex_set[prefix_id].get_size() == schedule->get_break_size(prefix_id)) {
                is_zero = true;
                break;
            }
        }
        if (is_zero)
            continue;
        GPU_pattern_matching_func<depth + 1>(schedule, vertex_set, local_ans, edge, vertex);
    }
}

    template <>
__device__ void GPU_pattern_matching_func<MAX_DEPTH>(const GPUSchedule* schedule, GPUVertexSet* vertex_set, unsigned long long& local_ans, uint32_t *edge, e_index_t *vertex)
{
    // assert(false);
}

/**
 * @note `buffer_size`实际上是每个节点的最大邻居数量，而非所用空间大小
 */
__global__ void gpu_pattern_matching(uint32_t edge_num, uint32_t buffer_size, uint32_t *edge_from, uint32_t *edge, e_index_t *vertex, uint32_t *tmp, const GPUSchedule* schedule) {
    __shared__ unsigned int block_edge_idx[WARPS_PER_BLOCK];
    //之后考虑把tmp buffer都放到shared里来（如果放得下）
    extern __shared__ GPUVertexSet block_vertex_set[];
    
    int num_prefixes = schedule->get_total_prefix_num();
    int num_vertex_sets_per_warp = num_prefixes + 2;

    int wid = threadIdx.x / THREADS_PER_WARP; // warp id within the block
    int lid = threadIdx.x % THREADS_PER_WARP; // lane id
    int global_wid = blockIdx.x * WARPS_PER_BLOCK + wid; // global warp id
    unsigned int &edge_idx = block_edge_idx[wid];
    GPUVertexSet *vertex_set = block_vertex_set + wid * num_vertex_sets_per_warp;

    if (lid == 0) {
        edge_idx = 0;
        uint32_t offset = buffer_size * global_wid * num_vertex_sets_per_warp;
        for (int i = 0; i < num_vertex_sets_per_warp; ++i)
        {
            vertex_set[i].set_data_ptr(tmp + offset); // 注意这是个指针+整数运算，自带*4
            offset += buffer_size;
        }
    }
    // GPUVertexSet& subtraction_set = vertex_set[num_prefixes];
    // GPUVertexSet& tmp_set = vertex_set[num_prefixes + 1];

    __threadfence_block(); //之后考虑把所有的syncthreads都改成syncwarp


    uint32_t v0, v1;
    long long l, r;

    unsigned long long sum = 0;

    while (true) {
        if (lid == 0) {
            //if(++edgeI >= edgeEnd) { //这个if语句应该是每次都会发生吧？（是的
                edge_idx = atomicAdd(&dev_cur_edge, 1);
                //edgeEnd = min(edge_num, edgeI + 1); //这里不需要原子读吗
                // unsigned int i = edge_idx;
                // if (i < edge_num)
                // {
                //     subtraction_set.init();
                //     subtraction_set.push_back(edge_from[i]);
                //     subtraction_set.push_back(edge[i]);
                // }
            //}
        }

        __threadfence_block();

        unsigned int i = edge_idx;
        if(i >= edge_num) break;
    
    // for edge in E
        v0 = edge_from[i];
        v1 = edge[i];

        bool is_zero = false;
        get_edge_index(v0, l, r);
        for (int prefix_id = schedule->get_last(0); prefix_id != -1; prefix_id = schedule->get_next(prefix_id)) {
            vertex_set[prefix_id].build_vertex_set(schedule, vertex_set, &edge[l], r - l, prefix_id);
            // if(lid == 0){
            //     printf("edge:%d id:%d , size: %d\n",i, prefix_id, vertex_set[prefix_id].get_size());
            // }
            // vertex_set[prefix_id].get_size();
        }

        //目前只考虑pattern size>2的情况
        // start v1, depth = 1
        if (schedule->get_restrict_last(1) != -1 && v0 <= v1)
            continue;

        // v_0 > v_1, 合理
        
        get_edge_index(v1, l, r);
        for (int prefix_id = schedule->get_last(1); prefix_id != -1; prefix_id = schedule->get_next(prefix_id))
        {
            vertex_set[prefix_id].build_vertex_set(schedule, vertex_set, &edge[l], r - l, prefix_id);
            if (vertex_set[prefix_id].get_size() == 0 && prefix_id < schedule->get_basic_prefix_num()) {
                is_zero = true;
                break;
            }
            // if(lid == 0){
            //     printf("edge:%d id:%d , size: %d\n",i, prefix_id, vertex_set[prefix_id].get_size());
            // }
        }
        if (is_zero)
            continue;
        
        unsigned long long local_sum = 0; // local sum (corresponding to an edge index)
        GPU_pattern_matching_func<2>(schedule, vertex_set, local_sum, edge, vertex);
        // GPU_pattern_matching_aggressive_func(schedule, vertex_set, subtraction_set, tmp_set, local_sum, 2, edge, vertex);
        sum += local_sum;
    }



    if (lid == 0) {
        atomicAdd(&dev_sum, sum);
    }
}

void pattern_matching_init(Graph *g, const Schedule_IEP& schedule_iep) {
    printf("basic prefix %d, total prefix %d\n", schedule_iep.get_basic_prefix_num(), schedule_iep.get_total_prefix_num());

    int num_blocks = 1024;
    int num_total_warps = num_blocks * WARPS_PER_BLOCK;

    size_t size_edge = g->e_cnt * sizeof(v_index_t);
    size_t size_vertex = (g->v_cnt + 1) * sizeof(e_index_t);
    size_t size_tmp = VertexSet::max_intersection_size * sizeof(uint32_t) * num_total_warps * (schedule_iep.get_total_prefix_num() + 2); //prefix + subtraction + tmp

    schedule_iep.print_schedule();
    uint32_t *edge_from = new uint32_t[g->e_cnt];
    for(uint32_t i = 0; i < g->v_cnt; ++i)
        for(e_index_t j = g->vertex[i]; j < g->vertex[i+1]; ++j)
            edge_from[j] = i;

    tmpTime.check(); 

    uint32_t *dev_edge;
    uint32_t *dev_edge_from;
    e_index_t *dev_vertex;
    uint32_t *dev_tmp;

    gpuErrchk( cudaMalloc((void**)&dev_edge, size_edge));
    gpuErrchk( cudaMalloc((void**)&dev_edge_from, size_edge));
    gpuErrchk( cudaMalloc((void**)&dev_vertex, size_vertex));
    gpuErrchk( cudaMalloc((void**)&dev_tmp, size_tmp));

    gpuErrchk( cudaMemcpy(dev_edge, g->edge, size_edge, cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy(dev_edge_from, edge_from, size_edge, cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy(dev_vertex, g->vertex, size_vertex, cudaMemcpyHostToDevice));

    unsigned long long sum = 0;
    gpuErrchk( cudaMemcpyToSymbol(dev_sum, &sum, sizeof(sum)));
    unsigned int cur_edge = 0;
    gpuErrchk( cudaMemcpyToSymbol(dev_cur_edge, &cur_edge, sizeof(cur_edge)));

    //memcpy schedule
    GPUSchedule* dev_schedule;
    gpuErrchk( cudaMallocManaged((void**)&dev_schedule, sizeof(GPUSchedule)));
    //dev_schedule->transform_in_exclusion_optimize_group_val(schedule);
    int schedule_size = schedule_iep.get_size();
    int max_prefix_num = schedule_size * (schedule_size - 1) / 2;
    
    bool *only_need_size = new bool[max_prefix_num];
    for(int i = 0; i < max_prefix_num; ++i)
        only_need_size[i] = schedule_iep.get_prefix_only_need_size(i);

    int in_exclusion_optimize_vertex_id_size = schedule_iep.in_exclusion_optimize_vertex_id.size();
    int in_exclusion_optimize_array_size  = schedule_iep.in_exclusion_optimize_coef.size();

    assert(in_exclusion_optimize_array_size == schedule_iep.in_exclusion_optimize_coef.size());
    assert(in_exclusion_optimize_array_size == schedule_iep.in_exclusion_optimize_flag.size());

    printf("array size %d\n", in_exclusion_optimize_array_size);
    fflush(stdout);

    int* in_exclusion_optimize_vertex_id = new int[in_exclusion_optimize_vertex_id_size];
    bool* in_exclusion_optimize_vertex_flag = new bool[in_exclusion_optimize_vertex_id_size];
    int* in_exclusion_optimize_vertex_coef = new int[in_exclusion_optimize_vertex_id_size];
    
    int* in_exclusion_optimize_coef = new int[in_exclusion_optimize_array_size];
    bool* in_exclusion_optimize_flag = new bool[in_exclusion_optimize_array_size];
    int* in_exclusion_optimize_ans_pos = new int[in_exclusion_optimize_array_size];

    for(int i = 0; i < in_exclusion_optimize_vertex_id_size; ++i) {
        in_exclusion_optimize_vertex_id[i] = schedule_iep.in_exclusion_optimize_vertex_id[i];
        in_exclusion_optimize_vertex_flag[i] = schedule_iep.in_exclusion_optimize_vertex_flag[i];
        in_exclusion_optimize_vertex_coef[i] = schedule_iep.in_exclusion_optimize_vertex_coef[i];
    }

    for(int i = 0; i < in_exclusion_optimize_array_size; ++i) {
        in_exclusion_optimize_coef[i] = schedule_iep.in_exclusion_optimize_coef[i];
        in_exclusion_optimize_flag[i] = schedule_iep.in_exclusion_optimize_flag[i];
        in_exclusion_optimize_ans_pos[i] = schedule_iep.in_exclusion_optimize_ans_pos[i];
    }

    if (in_exclusion_optimize_vertex_id_size > 0) {
        gpuErrchk( cudaMallocManaged((void**)&dev_schedule->in_exclusion_optimize_vertex_id, sizeof(int) * in_exclusion_optimize_vertex_id_size));
        gpuErrchk( cudaMemcpy(dev_schedule->in_exclusion_optimize_vertex_id, in_exclusion_optimize_vertex_id, sizeof(int) * in_exclusion_optimize_vertex_id_size, cudaMemcpyHostToDevice));
        
        gpuErrchk( cudaMallocManaged((void**)&dev_schedule->in_exclusion_optimize_vertex_flag, sizeof(bool) * in_exclusion_optimize_vertex_id_size));
        gpuErrchk( cudaMemcpy(dev_schedule->in_exclusion_optimize_vertex_flag, in_exclusion_optimize_vertex_flag, sizeof(bool) * in_exclusion_optimize_vertex_id_size, cudaMemcpyHostToDevice));
        
        gpuErrchk( cudaMallocManaged((void**)&dev_schedule->in_exclusion_optimize_vertex_coef, sizeof(int) * in_exclusion_optimize_vertex_id_size));
        gpuErrchk( cudaMemcpy(dev_schedule->in_exclusion_optimize_vertex_coef, in_exclusion_optimize_vertex_coef, sizeof(int) * in_exclusion_optimize_vertex_id_size, cudaMemcpyHostToDevice));
    }

    if (in_exclusion_optimize_array_size > 0)
    {
        gpuErrchk( cudaMallocManaged((void**)&dev_schedule->in_exclusion_optimize_coef, sizeof(int) * in_exclusion_optimize_array_size));
        gpuErrchk( cudaMemcpy(dev_schedule->in_exclusion_optimize_coef, in_exclusion_optimize_coef, sizeof(int) * in_exclusion_optimize_array_size, cudaMemcpyHostToDevice));

        gpuErrchk( cudaMallocManaged((void**)&dev_schedule->in_exclusion_optimize_flag, sizeof(bool) * in_exclusion_optimize_array_size));
        gpuErrchk( cudaMemcpy(dev_schedule->in_exclusion_optimize_flag, in_exclusion_optimize_flag, sizeof(bool) * in_exclusion_optimize_array_size, cudaMemcpyHostToDevice));
        
        gpuErrchk( cudaMallocManaged((void**)&dev_schedule->in_exclusion_optimize_ans_pos, sizeof(int) * in_exclusion_optimize_array_size));
        gpuErrchk( cudaMemcpy(dev_schedule->in_exclusion_optimize_ans_pos, in_exclusion_optimize_ans_pos, sizeof(int) * in_exclusion_optimize_array_size, cudaMemcpyHostToDevice));
    }

    gpuErrchk( cudaMallocManaged((void**)&dev_schedule->adj_mat, sizeof(int) * schedule_size * schedule_size));
    gpuErrchk( cudaMemcpy(dev_schedule->adj_mat, schedule_iep.get_adj_mat_ptr(), sizeof(int) * schedule_size * schedule_size, cudaMemcpyHostToDevice));

    gpuErrchk( cudaMallocManaged((void**)&dev_schedule->father_prefix_id, sizeof(int) * max_prefix_num));
    gpuErrchk( cudaMemcpy(dev_schedule->father_prefix_id, schedule_iep.get_father_prefix_id_ptr(), sizeof(int) * max_prefix_num, cudaMemcpyHostToDevice));

    gpuErrchk( cudaMallocManaged((void**)&dev_schedule->last, sizeof(int) * schedule_size));
    gpuErrchk( cudaMemcpy(dev_schedule->last, schedule_iep.get_last_ptr(), sizeof(int) * schedule_size, cudaMemcpyHostToDevice));

    gpuErrchk( cudaMallocManaged((void**)&dev_schedule->next, sizeof(int) * max_prefix_num));
    gpuErrchk( cudaMemcpy(dev_schedule->next, schedule_iep.get_next_ptr(), sizeof(int) * max_prefix_num, cudaMemcpyHostToDevice));
    
    gpuErrchk( cudaMallocManaged((void**)&dev_schedule->only_need_size, sizeof(bool) * max_prefix_num));
    gpuErrchk( cudaMemcpy(dev_schedule->only_need_size, only_need_size, sizeof(bool) * max_prefix_num, cudaMemcpyHostToDevice));
    
    gpuErrchk( cudaMallocManaged((void**)&dev_schedule->break_size, sizeof(int) * max_prefix_num));
    gpuErrchk( cudaMemcpy(dev_schedule->break_size, schedule_iep.get_break_size_ptr(), sizeof(int) * max_prefix_num, cudaMemcpyHostToDevice));

    gpuErrchk( cudaMallocManaged((void**)&dev_schedule->loop_set_prefix_id, sizeof(int) * schedule_size));
    gpuErrchk( cudaMemcpy(dev_schedule->loop_set_prefix_id, schedule_iep.get_loop_set_prefix_id_ptr(), sizeof(int) * schedule_size, cudaMemcpyHostToDevice));

    gpuErrchk( cudaMallocManaged((void**)&dev_schedule->restrict_last, sizeof(int) * schedule_size));
    gpuErrchk( cudaMemcpy(dev_schedule->restrict_last, schedule_iep.get_restrict_last_ptr(), sizeof(int) * schedule_size, cudaMemcpyHostToDevice));
    
    gpuErrchk( cudaMallocManaged((void**)&dev_schedule->restrict_next, sizeof(int) * max_prefix_num));
    gpuErrchk( cudaMemcpy(dev_schedule->restrict_next, schedule_iep.get_restrict_next_ptr(), sizeof(int) * max_prefix_num, cudaMemcpyHostToDevice));
    
    gpuErrchk( cudaMallocManaged((void**)&dev_schedule->restrict_index, sizeof(int) * max_prefix_num));
    gpuErrchk( cudaMemcpy(dev_schedule->restrict_index, schedule_iep.get_restrict_index_ptr(), sizeof(int) * max_prefix_num, cudaMemcpyHostToDevice));

    dev_schedule->in_exclusion_optimize_array_size = in_exclusion_optimize_array_size;
    dev_schedule->in_exclusion_optimize_vertex_id_size = in_exclusion_optimize_vertex_id_size;
    dev_schedule->size = schedule_iep.get_size();
    dev_schedule->total_prefix_num = schedule_iep.get_total_prefix_num();
    dev_schedule->basic_prefix_num = schedule_iep.get_basic_prefix_num();
    dev_schedule->total_restrict_num = schedule_iep.get_total_restrict_num();
    dev_schedule->in_exclusion_optimize_num = schedule_iep.get_in_exclusion_optimize_num();
    //dev_schedule->k_val = schedule.get_k_val();

    printf("schedule_iep.prefix_num: %d\n", schedule_iep.get_total_prefix_num());
    printf("shared memory for vertex set per block: %ld bytes\n", 
        (schedule_iep.get_total_prefix_num() + 2) * WARPS_PER_BLOCK * sizeof(GPUVertexSet) + in_exclusion_optimize_vertex_id_size * WARPS_PER_BLOCK * sizeof(int));

    tmpTime.print("Prepare time cost");
    tmpTime.check();

    uint32_t buffer_size = VertexSet::max_intersection_size;
    uint32_t block_shmem_size = (schedule_iep.get_total_prefix_num() + 2) * WARPS_PER_BLOCK * sizeof(GPUVertexSet) + in_exclusion_optimize_vertex_id_size * WARPS_PER_BLOCK * sizeof(int);
    dev_schedule->ans_array_offset = block_shmem_size - in_exclusion_optimize_vertex_id_size * WARPS_PER_BLOCK * sizeof(int);
    // 注意：此处没有错误，buffer_size代指每个顶点集所需的int数目，无需再乘sizeof(uint32_t)，但是否考虑对齐？
    //因为目前用了managed开内存，所以第一次运行kernel会有一定额外开销，考虑运行两次，第一次作为warmup
    
    int max_active_blocks_per_sm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks_per_sm, gpu_pattern_matching, THREADS_PER_BLOCK, block_shmem_size);
    printf("max number of active warps per SM: %d\n", max_active_blocks_per_sm * WARPS_PER_BLOCK);
    
    gpu_pattern_matching<<<num_blocks, THREADS_PER_BLOCK, block_shmem_size>>>
        (g->e_cnt, buffer_size, dev_edge_from, dev_edge, dev_vertex, dev_tmp, dev_schedule);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );


    gpuErrchk( cudaMemcpyFromSymbol(&sum, dev_sum, sizeof(sum)) );

    sum /= schedule_iep.get_in_exclusion_optimize_redundancy();

    printf("%d\n",schedule_iep.get_in_exclusion_optimize_redundancy());
    
    #ifdef PRINT_ANS_TO_FILE
    freopen("1.out", "w", stdout);
    printf("count %llu\n", sum);
    fclose(stdout);
    #endif
    printf("count %llu\n", sum);
    tmpTime.print("Counting time cost");
    //之后需要加上cudaFree

    // 尝试释放一些内存
    gpuErrchk(cudaFree(dev_edge));
    gpuErrchk(cudaFree(dev_edge_from));
    gpuErrchk(cudaFree(dev_vertex));
    gpuErrchk(cudaFree(dev_tmp));

    gpuErrchk(cudaFree(dev_schedule->adj_mat));
    gpuErrchk(cudaFree(dev_schedule->father_prefix_id));
    gpuErrchk(cudaFree(dev_schedule->last));
    gpuErrchk(cudaFree(dev_schedule->next));
    gpuErrchk(cudaFree(dev_schedule->loop_set_prefix_id));
    gpuErrchk(cudaFree(dev_schedule->restrict_last));
    gpuErrchk(cudaFree(dev_schedule->restrict_next));
    gpuErrchk(cudaFree(dev_schedule->restrict_index));

    gpuErrchk(cudaFree(dev_schedule->in_exclusion_optimize_vertex_id));
    gpuErrchk(cudaFree(dev_schedule->in_exclusion_optimize_coef));
    gpuErrchk(cudaFree(dev_schedule->in_exclusion_optimize_flag));

    gpuErrchk(cudaFree(dev_schedule));

    delete[] edge_from;
    delete[] in_exclusion_optimize_vertex_id;
    delete[] in_exclusion_optimize_coef;
    delete[] in_exclusion_optimize_flag;
    delete[] only_need_size;
}


int main(int argc,char *argv[]) {
    Graph *g;
    DataLoader D;

    using std::chrono::system_clock;
    auto t1 = system_clock::now();
    
    if(argc != 3) {
        printf("usage: %s graph_file clique_size", argv[0]);
        return 0;
    }

    // 读入 pattern size
    int pattern_size = atoi(argv[2]);

    Pattern p(pattern_size);

    for(int i = 0; i < pattern_size; i++) {
        for(int j = i + 1; j < pattern_size; j++) {
            p.add_edge(i, j);
        }
    }

    DataType type = DataType::Patents;

    bool ok = D.load_data(g, type, argv[1]);

    if (!ok) {
        printf("data load failure :-(\n");
        return 0;
    }


    reduce_edges_for_clique(*g);
    // warm up GPU
    cudaFree(0);


    auto t2 = system_clock::now();
    auto load_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    printf("Load data success! time: %g seconds\n", load_time.count() / 1.0e6);
    fflush(stdout);

    allTime.check();


    printf("pattern = \n");
    p.print();
    printf("max intersection size %d\n", VertexSet::max_intersection_size);
    bool is_pattern_valid;
    bool use_in_exclusion_optimize = true;
    Schedule_IEP schedule_iep(p, is_pattern_valid, 1, 1, use_in_exclusion_optimize, g->v_cnt, g->e_cnt, g->tri_cnt);
    schedule_iep.set_in_exclusion_optimize_redundancy(1);

    if (!is_pattern_valid) {
        printf("pattern is invalid!\n");
        return 0;
    }

    pattern_matching_init(g, schedule_iep);

    allTime.print("Total time cost");

    return 0;
}