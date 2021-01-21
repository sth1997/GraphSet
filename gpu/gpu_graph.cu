#include <graph.h>
#include <dataloader.h>
#include <vertex_set.h>
#include <common.h>

#include <cassert>
#include <cstring>
#include <cstdint>
#include <cmath>

#include <limits> 
#include <algorithm>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <sys/time.h>
#include <chrono>

constexpr int THREADS_PER_BLOCK = 256;
constexpr int THREADS_PER_WARP = 32;
constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / THREADS_PER_WARP;

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

#define unlikely(x) __builtin_expect(!!(x), 0)

template <typename T>
__device__ inline void dev_swap(T& a, T& b)
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

/**
 * @todo 增加处理IEP造成的重复计数
 */
class GPUSchedule {
public:
    __host__ void transform_in_exclusion_optimize_group_val(const Schedule& schedule)
    {
        /** @todo 注意当容斥优化无法使用时，内存分配会失败。需要修正 */
        int in_exclusion_optimize_num = schedule.get_in_exclusion_optimize_num();
        gpuErrchk( cudaMallocManaged((void**)&in_exclusion_optimize_val, sizeof(int) * schedule.in_exclusion_optimize_val.size()));
        for (auto val : schedule.in_exclusion_optimize_val)
            in_exclusion_optimize_val[in_exclusion_optimize_val_size++] = val;
        in_exclusion_optimize_val_size = schedule.in_exclusion_optimize_val.size();
        
        //这部分有太多重复访存操作了（比如循环中的.data[i].data[j]，直接用一个tmp指针就行了），之后考虑优化掉（不过感觉O3会帮忙自动优化的）
        in_exclusion_optimize_group.size = schedule.in_exclusion_optimize_group.size();
        gpuErrchk( cudaMallocManaged((void**)&in_exclusion_optimize_group.data, sizeof(GPUGroupDim1) * in_exclusion_optimize_group.size));
        for (int i = 0; i < schedule.in_exclusion_optimize_group.size(); ++i)
        {
            in_exclusion_optimize_group.data[i].size = schedule.in_exclusion_optimize_group[i].size();
            gpuErrchk( cudaMallocManaged((void**)&in_exclusion_optimize_group.data[i].data, sizeof(GPUGroupDim2) * in_exclusion_optimize_group.data[i].size));
            for (int j = 0; j < schedule.in_exclusion_optimize_group[i].size(); ++j)
            {
                in_exclusion_optimize_group.data[i].data[j].size = schedule.in_exclusion_optimize_group[i][j].size();
                gpuErrchk( cudaMallocManaged((void**)&in_exclusion_optimize_group.data[i].data[j].data, sizeof(int) * in_exclusion_optimize_group.data[i].data[j].size));
                for (int k = 0; k < schedule.in_exclusion_optimize_group[i][j].size(); ++k)
                    in_exclusion_optimize_group.data[i].data[j].data[k] = schedule.in_exclusion_optimize_group[i][j][k];
            }
        }
    }

    __host__ void init_from(const Schedule& schedule)
    {
        transform_in_exclusion_optimize_group_val(schedule);

        int schedule_size = schedule.get_size();
        int max_prefix_num = schedule_size * (schedule_size - 1) / 2;

        gpuErrchk( cudaMallocManaged(&adj_mat, sizeof(int) * schedule_size * schedule_size));
        gpuErrchk( cudaMemcpy(adj_mat, schedule.get_adj_mat_ptr(), sizeof(int) * schedule_size * schedule_size, cudaMemcpyHostToDevice));

        gpuErrchk( cudaMallocManaged(&father_prefix_id, sizeof(int) * max_prefix_num));
        gpuErrchk( cudaMemcpy(father_prefix_id, schedule.get_father_prefix_id_ptr(), sizeof(int) * max_prefix_num, cudaMemcpyHostToDevice));

        gpuErrchk( cudaMallocManaged(&last, sizeof(int) * schedule_size));
        gpuErrchk( cudaMemcpy(last, schedule.get_last_ptr(), sizeof(int) * schedule_size, cudaMemcpyHostToDevice));

        gpuErrchk( cudaMallocManaged(&next, sizeof(int) * max_prefix_num));
        gpuErrchk( cudaMemcpy(next, schedule.get_next_ptr(), sizeof(int) * max_prefix_num, cudaMemcpyHostToDevice));

        gpuErrchk( cudaMallocManaged(&loop_set_prefix_id, sizeof(int) * schedule_size));
        gpuErrchk( cudaMemcpy(loop_set_prefix_id, schedule.get_loop_set_prefix_id_ptr(), sizeof(int) * schedule_size, cudaMemcpyHostToDevice));

        gpuErrchk( cudaMallocManaged(&restrict_last, sizeof(int) * schedule_size));
        gpuErrchk( cudaMemcpy(restrict_last, schedule.get_restrict_last_ptr(), sizeof(int) * schedule_size, cudaMemcpyHostToDevice));
    
        gpuErrchk( cudaMallocManaged(&restrict_next, sizeof(int) * max_prefix_num));
        gpuErrchk( cudaMemcpy(restrict_next, schedule.get_restrict_next_ptr(), sizeof(int) * max_prefix_num, cudaMemcpyHostToDevice));
    
        gpuErrchk( cudaMallocManaged(&restrict_index, sizeof(int) * max_prefix_num));
        gpuErrchk( cudaMemcpy(restrict_index, schedule.get_restrict_index_ptr(), sizeof(int) * max_prefix_num, cudaMemcpyHostToDevice));

        size = schedule.get_size();
        total_prefix_num = schedule.get_total_prefix_num();
        total_restrict_num = schedule.get_total_restrict_num();
        in_exclusion_optimize_num = schedule.get_in_exclusion_optimize_num();
        k_val = schedule.get_k_val();
    }

    __host__ void release()
    {
        gpuErrchk(cudaFree(in_exclusion_optimize_val));
        for (int i = 0; i < in_exclusion_optimize_group.size; ++i) {
            for (int j = 0; j < in_exclusion_optimize_group.data[i].size; ++j)
                gpuErrchk(cudaFree(in_exclusion_optimize_group.data[i].data[j].data));
            gpuErrchk(cudaFree(in_exclusion_optimize_group.data[i].data));
        }
        gpuErrchk(cudaFree(in_exclusion_optimize_group.data));

        gpuErrchk(cudaFree(adj_mat));
        gpuErrchk(cudaFree(father_prefix_id));
        gpuErrchk(cudaFree(last));
        gpuErrchk(cudaFree(next));
        gpuErrchk(cudaFree(loop_set_prefix_id));
        gpuErrchk(cudaFree(restrict_last));
        gpuErrchk(cudaFree(restrict_next));
        gpuErrchk(cudaFree(restrict_index));
    }

    inline __device__ int get_total_prefix_num() const { return total_prefix_num;}
    inline __device__ int get_father_prefix_id(int prefix_id) const { return father_prefix_id[prefix_id];}
    inline __device__ int get_loop_set_prefix_id(int loop) const { return loop_set_prefix_id[loop];}
    inline __device__ int get_size() const { return size;}
    inline __device__ int get_last(int i) const { return last[i];}
    inline __device__ int get_next(int i) const { return next[i];}
    inline __device__ int get_in_exclusion_optimize_num() const { return in_exclusion_optimize_num;}
    inline __device__ int get_total_restrict_num() const { return total_restrict_num;}
    inline __device__ int get_restrict_last(int i) const { return restrict_last[i];}
    inline __device__ int get_restrict_next(int i) const { return restrict_next[i];}
    inline __device__ int get_restrict_index(int i) const { return restrict_index[i];}
    inline __device__ int get_k_val() const { return k_val;} // see below (the k_val's definition line) before using this function

    int* adj_mat;
    int* father_prefix_id;
    int* last;
    int* next;
    int* loop_set_prefix_id;
    int* restrict_last;
    int* restrict_next;
    int* restrict_index;
    int* in_exclusion_optimize_val;
    GPUGroupDim0 in_exclusion_optimize_group;
    int in_exclusion_optimize_val_size;
    int size;
    int total_prefix_num;
    int total_restrict_num;
    int in_exclusion_optimize_num;
    int k_val;
};

// __device__ void intersection1(uint32_t *tmp, uint32_t *lbases, uint32_t *rbases, uint32_t ln, uint32_t rn, uint32_t* p_tmp_size);
__device__ void intersection2(uint32_t *tmp, const uint32_t *lbases, const uint32_t *rbases, uint32_t ln, uint32_t rn, uint32_t* p_tmp_size);
static __device__ int do_intersection(uint32_t*, const uint32_t*, const uint32_t*, int, int);

class GPUVertexSet
{
public:
    __device__ GPUVertexSet()
    {
        size = 0;
        data = NULL;
    }
    __device__ uint32_t get_size() const { return size;}
    __device__ uint32_t& operator[](int i) { return data[i]; }
    __device__ uint32_t get_data(int i) const { return data[i];}
    __device__ void push_back(uint32_t val) { data[size++] = val;}
    __device__ void pop_back() { --size;}
    __device__ uint32_t get_last() const {return data[size - 1];}
    __device__ void set_data_ptr(uint32_t* ptr) { data = ptr;}
    __device__ uint32_t* get_data_ptr() const { return data;}
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
            intersection2(this->data, vertex_set[father_id].get_data_ptr(), input_data, vertex_set[father_id].get_size(), input_size, &this->size);
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

class GPUSubtractionSet {
public:
    static constexpr int MAX_NR_ELEMENTS = 5;

    __device__ void init() { size = 0; }
    __device__ uint32_t get_size() const { return size; }
    __device__ uint32_t get_data(int i) const { return data[i]; }
    __device__ uint32_t& operator[](int i) { return data[i]; }
    __device__ const uint32_t* get_data_ptr() const { return data; }
    __device__ void push_back(uint32_t v) { data[size++] = v; }
    __device__ void pop_back() { --size; }
    __device__ bool has_data(uint32_t v) const
    {
        for (int i = 0; i < size; ++i) // 注意：这里不用二分，调用它的是较小的无序集合
            if (data[i] == v)
                return true;
        return false;
    }
    __device__ bool has_data_unrolled(uint32_t v) const
    {
        #pragma unroll
        for (int i = 0; i < MAX_NR_ELEMENTS; ++i) {
            if (i >= size) break;
            if (data[i] == v) return true;
        }
        return false;
    }

private:
    uint32_t size, data[MAX_NR_ELEMENTS]; 
};

__device__ unsigned long long dev_sum = 0;
__device__ unsigned int dev_cur_edge = 0;

#include "set_ops.cuh"

/**
 * wrapper of search based intersection `do_intersection`
 * 
 * 注意：不能进行in-place操作。若想原地操作则应当把交换去掉。
 */
__device__ void intersection2(uint32_t *tmp, const uint32_t *lbases, const uint32_t *rbases, uint32_t ln, uint32_t rn, uint32_t* p_tmp_size)
{
    // make sure ln <= rn
    if (ln > rn) {
        dev_swap(ln, rn);
        dev_swap(lbases, rbases);
    }
    /**
     * @todo 考虑ln < rn <= 32时，每个线程在lbases里面找rbases的一个元素可能会更快
     */

    uint32_t intersection_size = do_intersection(tmp, lbases, rbases, ln, rn);

    if (threadIdx.x % THREADS_PER_WARP == 0)
        *p_tmp_size = intersection_size;
    __threadfence_block();
}

/**
 * @brief 递归版本的pattern matching主函数。
 * @note 调用处初始深度为2（已经匹配了一条边对应的两个点）
 */
__device__ void GPU_pattern_matching_aggressive_func(const GPUSchedule* schedule, GPUVertexSet* vertex_set, GPUSubtractionSet& subtraction_set,
    GPUVertexSet& tmp_set, unsigned long long& local_ans, int depth, uint32_t *edge, const uint32_t *vertex)
{
    int loop_set_prefix_id = schedule->get_loop_set_prefix_id(depth);
    int loop_size = vertex_set[loop_set_prefix_id].get_size();
    if (loop_size <= 0)
        return;

    uint32_t* loop_data_ptr = vertex_set[loop_set_prefix_id].get_data_ptr();

    if( depth == schedule->get_size() - schedule->get_in_exclusion_optimize_num())
    {
        int in_exclusion_optimize_num = schedule->get_in_exclusion_optimize_num();
        //int* loop_set_prefix_ids[ in_exclusion_optimize_num ];
        int loop_set_prefix_ids[8]; // @todo 偷懒用了static，之后考虑改成dynamic
        // @todo 这里有硬编码的数字，之后考虑修改
        loop_set_prefix_ids[0] = loop_set_prefix_id;
        for(int i = 1; i < in_exclusion_optimize_num; ++i)
            loop_set_prefix_ids[i] = schedule->get_loop_set_prefix_id( depth + i );

        for(int optimize_rank = 0; optimize_rank < schedule->in_exclusion_optimize_group.size; ++optimize_rank) {
            const GPUGroupDim1& cur_graph = schedule->in_exclusion_optimize_group.data[optimize_rank];
            long long val = schedule->in_exclusion_optimize_val[optimize_rank];

            for(int cur_graph_rank = 0; cur_graph_rank < cur_graph.size; ++cur_graph_rank) {
                if(cur_graph.data[cur_graph_rank].size == 1) {
                    int id = loop_set_prefix_ids[cur_graph.data[cur_graph_rank].data[0]];
                    //val = val * unordered_subtraction_size(vertex_set[id], subtraction_set);
                    val *= set_difference_size(vertex_set[id], subtraction_set);
                }
                else {
                    int id = loop_set_prefix_ids[cur_graph.data[cur_graph_rank].data[0]];
                    tmp_set.copy_from(vertex_set[id]);

                    for(int i = 1; i < cur_graph.data[cur_graph_rank].size; ++i) {
                        int id = loop_set_prefix_ids[cur_graph.data[cur_graph_rank].data[i]];
                        tmp_set.intersection_with(vertex_set[id]);
                    }
                    
                    val *= set_difference_size(tmp_set, subtraction_set);
                }
                if (val == 0)
                    break;
            }

            local_ans += val;
        }
        return;
    }

    // 无容斥优化的最后一层
    if (depth == schedule->get_size() - 1)
    {
        //TODO
        assert(false);

        //if (threadIdx.x == 0)
        //    local_ans += val;
    }

    uint32_t min_vertex = 0xffffffff;
    for (int i = schedule->get_restrict_last(depth); i != -1; i = schedule->get_restrict_next(i))
        if (min_vertex > subtraction_set.get_data(schedule->get_restrict_index(i)))
            min_vertex = subtraction_set.get_data(schedule->get_restrict_index(i));
    for (int i = 0; i < loop_size; ++i)
    {
        uint32_t v = loop_data_ptr[i];
        if (min_vertex <= v)
            break;
        if (subtraction_set.has_data(v))
            continue;
        unsigned int l, r;
        get_edge_index(v, l, r);
        bool is_zero = false;
        for (int prefix_id = schedule->get_last(depth); prefix_id != -1; prefix_id = schedule->get_next(prefix_id))
        {
            vertex_set[prefix_id].build_vertex_set(schedule, vertex_set, &edge[l], r - l, prefix_id);
            if (vertex_set[prefix_id].get_size() == 0) {
                is_zero = true;
                break;
            }
        }
        if (is_zero)
            continue;
        if(threadIdx.x % THREADS_PER_WARP == 0)
            subtraction_set.push_back(v);
        __threadfence_block();
        GPU_pattern_matching_aggressive_func(schedule, vertex_set, subtraction_set, tmp_set, local_ans, depth + 1, edge, vertex);
        if(threadIdx.x % THREADS_PER_WARP == 0)
            subtraction_set.pop_back();
        __threadfence_block();
    }
}


__device__ unsigned long long IEP_3_layer_v1(const GPUSchedule* schedule, GPUVertexSet* vertex_set, const GPUSubtractionSet& subtraction_set,
    GPUVertexSet& tmp_set, int depth)
{
    int i, j, k;
    i = schedule->get_loop_set_prefix_id(depth);
    j = schedule->get_loop_set_prefix_id(depth + 1);
    k = schedule->get_loop_set_prefix_id(depth + 2);
    // sort 3 sets according to their sizes
    if (vertex_set[k].get_size() < vertex_set[j].get_size())
        dev_swap(k, j);
    if (vertex_set[j].get_size() < vertex_set[i].get_size())
        dev_swap(j, i);
    if (vertex_set[k].get_size() < vertex_set[j].get_size())
        dev_swap(k, j);

    const GPUVertexSet &A = vertex_set[i];
    const GPUVertexSet &B = vertex_set[j];
    const GPUVertexSet &C = vertex_set[k];
    const GPUSubtractionSet &D = subtraction_set;
    uint32_t *buffer = tmp_set.get_data_ptr();

    int tmp_AB_size = do_intersection(buffer, A, B); // A ∩ B
    unsigned long long ABC_size = set_fused_op1(buffer, tmp_AB_size, C, D); // A ∩ B ∩ C - D
    // note: AB_size == |A \cap B|, ABC_size == |A \cap B \cap C - D|

    auto ret_sizes = set_fused_op2(A, B, C, D); // A ∩ C - D, B ∩ C - D
    unsigned long long AC_size = ret_sizes[0];
    unsigned long long BC_size = ret_sizes[1];
    // note: AC_size == |A \cap C - D|, BC_size == |B \cap C - D|

    int wid = threadIdx.x / THREADS_PER_WARP;
    int lid = threadIdx.x % THREADS_PER_WARP;
    __shared__ const uint32_t* block_A_ptrs[4 * WARPS_PER_BLOCK];
    __shared__ int  block_A_sizes[4 * WARPS_PER_BLOCK];
    const uint32_t **A_ptrs = block_A_ptrs + wid * 4;
    int *A_sizes = block_A_sizes + wid * 4;
    if (lid == 0) {
        A_ptrs[0] = A.get_data_ptr();
        A_ptrs[1] = B.get_data_ptr();
        A_ptrs[2] = C.get_data_ptr();
        A_ptrs[3] = buffer;
        A_sizes[0] = A.get_size();
        A_sizes[1] = B.get_size();
        A_sizes[2] = C.get_size();
        A_sizes[3] = tmp_AB_size;
    }
    __threadfence_block();

    ret_sizes = set_difference_size_compacted(A_ptrs, A_sizes, D, 4);
    unsigned long long A_size = ret_sizes[0];  // A - D
    unsigned long long B_size = ret_sizes[1];  // B - D
    unsigned long long C_size = ret_sizes[2];  // C - D
    unsigned long long AB_size = ret_sizes[3]; // A ∩ B - D
    return A_size * B_size * C_size - A_size * BC_size - B_size * AC_size - C_size * AB_size + ABC_size * 2; 
}

__device__ unsigned long long IEP_3_layer_v2(const GPUSchedule* schedule, GPUVertexSet* vertex_set, const GPUSubtractionSet& subtraction_set,
    GPUVertexSet& tmp_set, int depth)
{
    int i, j, k;
    i = schedule->get_loop_set_prefix_id(depth);
    j = schedule->get_loop_set_prefix_id(depth + 1);
    k = schedule->get_loop_set_prefix_id(depth + 2);
    // sort 3 sets according to their sizes
    if (vertex_set[k].get_size() < vertex_set[j].get_size())
        dev_swap(k, j);
    if (vertex_set[j].get_size() < vertex_set[i].get_size())
        dev_swap(j, i);
    if (vertex_set[k].get_size() < vertex_set[j].get_size())
        dev_swap(k, j);

    const GPUVertexSet &A = vertex_set[i];
    const GPUVertexSet &B = vertex_set[j];
    const GPUVertexSet &C = vertex_set[k];
    const GPUSubtractionSet &D = subtraction_set;
    uint32_t *buffer = tmp_set.get_data_ptr();

    int tmp_AB_size = do_intersection(buffer, A, B); // A ∩ B
    unsigned long long ABC_size = set_fused_op1(buffer, tmp_AB_size, C, D); // A ∩ B ∩ C - D

    auto ret_sizes = set_fused_op2(A, B, C, D); // A ∩ C - D, B ∩ C - D
    unsigned long long AC_size = ret_sizes[0];
    unsigned long long BC_size = ret_sizes[1];

    ret_sizes = set_difference_size_c2(buffer, tmp_AB_size, C, D);
    unsigned long long AB_size = ret_sizes[0]; // A ∩ B - D
    unsigned long long C_size  = ret_sizes[1]; // C - D

    ret_sizes = set_difference_size_c2(A, B, D);
    unsigned long long A_size = ret_sizes[0]; // A - D
    unsigned long long B_size = ret_sizes[1]; // B - D

    return A_size * B_size * C_size - A_size * BC_size - B_size * AC_size - C_size * AB_size + ABC_size * 2; 
}

__device__ unsigned long long IEP_3_layer_baseline(const GPUSchedule* schedule, GPUVertexSet* vertex_set, const GPUSubtractionSet& subtraction_set,
    GPUVertexSet& tmp_set, int depth)
{
    int i, j, k;
    i = schedule->get_loop_set_prefix_id(depth);
    j = schedule->get_loop_set_prefix_id(depth + 1);
    k = schedule->get_loop_set_prefix_id(depth + 2);
    // sort 3 sets according to their sizes
    if (vertex_set[k].get_size() < vertex_set[j].get_size())
        dev_swap(k, j);
    if (vertex_set[j].get_size() < vertex_set[i].get_size())
        dev_swap(j, i);
    if (vertex_set[k].get_size() < vertex_set[j].get_size())
        dev_swap(k, j);

    const GPUVertexSet &A = vertex_set[i];
    const GPUVertexSet &B = vertex_set[j];
    const GPUVertexSet &C = vertex_set[k];
    const GPUSubtractionSet &D = subtraction_set;
    uint32_t *buffer = tmp_set.get_data_ptr();

    int tmp_AB_size = do_intersection(buffer, A, B); // A ∩ B
    unsigned long long AB_size = set_difference_size(buffer, tmp_AB_size, D); // A ∩ B - D

    unsigned long long ABC_size = do_intersection(buffer, buffer, tmp_AB_size, C); // A ∩ B ∩ C
    ABC_size = set_difference_size(buffer, ABC_size, D); // A ∩ B ∩ C - D

    unsigned long long AC_size = do_intersection(buffer, A, C); // A ∩ C
    AC_size = set_difference_size(buffer, AC_size, D); // A ∩ C - D

    unsigned long long BC_size = do_intersection(buffer, B, C); // B ∩ C
    BC_size = set_difference_size(buffer, BC_size, D); // B ∩ C - D

    unsigned long long A_size = set_difference_size(A, D); // A - D
    unsigned long long B_size = set_difference_size(B, D); // B - D
    unsigned long long C_size = set_difference_size(C, D); // C - D
    return A_size * B_size * C_size - A_size * BC_size - B_size * AC_size - C_size * AB_size + ABC_size * 2; 
}

/**
 * @brief 最终层的容斥原理优化计算。
 */
__device__ inline void GPU_pattern_matching_final_in_exclusion(const GPUSchedule* schedule, GPUVertexSet* vertex_set, GPUSubtractionSet& subtraction_set,
    GPUVertexSet& tmp_set, unsigned long long& local_ans, int depth, uint32_t *edge, const uint32_t *vertex)
{
    int in_exclusion_optimize_num = schedule->get_in_exclusion_optimize_num();
    
    if (in_exclusion_optimize_num == 3) {
        local_ans += IEP_3_layer_v2(schedule, vertex_set, subtraction_set, tmp_set, depth);
        return;
    }

    /*
    //int* loop_set_prefix_ids[ in_exclusion_optimize_num ];
    __shared__ int loop_set_prefix_ids_block[8 * WARPS_PER_BLOCK];//偷懒用了static，之后考虑改成dynamic
    int *loop_set_prefix_ids = loop_set_prefix_ids_block + threadIdx.x / THREADS_PER_WARP * 8;
    // 这里有硬编码的数字，之后考虑修改。
    loop_set_prefix_ids[0] = schedule->get_loop_set_prefix_id(depth);
    for(int i = 1; i < in_exclusion_optimize_num; ++i)
        loop_set_prefix_ids[i] = schedule->get_loop_set_prefix_id( depth + i );

    for (int optimize_rank = 0; optimize_rank < schedule->in_exclusion_optimize_group.size; ++optimize_rank) {
        const GPUGroupDim1& cur_graph = schedule->in_exclusion_optimize_group.data[optimize_rank];
        long long val = schedule->in_exclusion_optimize_val[optimize_rank];

        for (int cur_graph_rank = 0; cur_graph_rank < cur_graph.size; ++cur_graph_rank) {
            if (cur_graph.data[cur_graph_rank].size == 1) {
                int id = loop_set_prefix_ids[cur_graph.data[cur_graph_rank].data[0]];
                //val = val * unordered_subtraction_size(vertex_set[id], subtraction_set);
                val *= set_difference_size(vertex_set[id], subtraction_set);
            } else {
                int id = loop_set_prefix_ids[cur_graph.data[cur_graph_rank].data[0]];
                tmp_set.copy_from(vertex_set[id]);

                for (int i = 1; i < cur_graph.data[cur_graph_rank].size; ++i) {
                    int id = loop_set_prefix_ids[cur_graph.data[cur_graph_rank].data[i]];
                    tmp_set.intersection_with(vertex_set[id]);
                }

                val *= set_difference_size(tmp_set, subtraction_set);
            }
            if (val == 0)
                break;
        }

        local_ans += val;
    }*/
}

constexpr int MAX_DEPTH = 5; // 非递归pattern matching支持的最大深度

template <int depth>
__device__ void GPU_pattern_matching_func(const GPUSchedule* schedule, GPUVertexSet* vertex_set, GPUSubtractionSet& subtraction_set,
    GPUVertexSet& tmp_set, unsigned long long& local_ans, uint32_t *edge, const uint32_t *vertex)
{
    int loop_set_prefix_id = schedule->get_loop_set_prefix_id(depth);
    int loop_size = vertex_set[loop_set_prefix_id].get_size();
    if (loop_size <= 0)
        return;

    if (depth == schedule->get_size() - schedule->get_in_exclusion_optimize_num()) {
        GPU_pattern_matching_final_in_exclusion(schedule, vertex_set, subtraction_set, tmp_set, local_ans, depth, edge, vertex);
        return;
    }

    uint32_t* loop_data_ptr = vertex_set[loop_set_prefix_id].get_data_ptr();
    uint32_t min_vertex = 0xffffffff;
    for (int i = schedule->get_restrict_last(depth); i != -1; i = schedule->get_restrict_next(i))
        if (min_vertex > subtraction_set.get_data(schedule->get_restrict_index(i)))
            min_vertex = subtraction_set.get_data(schedule->get_restrict_index(i));
    for (int i = 0; i < loop_size; ++i)
    {
        uint32_t v = loop_data_ptr[i];
        if (min_vertex <= v)
            break;
        if (subtraction_set.has_data(v))
            continue;
        unsigned int l, r;
        get_edge_index(v, l, r);
        bool is_zero = false;
        for (int prefix_id = schedule->get_last(depth); prefix_id != -1; prefix_id = schedule->get_next(prefix_id))
        {
            vertex_set[prefix_id].build_vertex_set(schedule, vertex_set, &edge[l], r - l, prefix_id);
            if (vertex_set[prefix_id].get_size() == 0) {
                is_zero = true;
                break;
            }
        }
        if (is_zero)
            continue;
        if (depth + 1 != MAX_DEPTH) {
            if (threadIdx.x % THREADS_PER_WARP == 0)
                subtraction_set.push_back(v);
            __threadfence_block();
        }
        GPU_pattern_matching_func<depth + 1>(schedule, vertex_set, subtraction_set, tmp_set, local_ans, edge, vertex);
        if (depth + 1 != MAX_DEPTH) {
            if (threadIdx.x % THREADS_PER_WARP == 0)
                subtraction_set.pop_back();
            __threadfence_block();
        }
    }
}

template <>
__device__ void GPU_pattern_matching_func<MAX_DEPTH>(const GPUSchedule* schedule, GPUVertexSet* vertex_set, GPUSubtractionSet& subtraction_set,
    GPUVertexSet& tmp_set, unsigned long long& local_ans, uint32_t *edge, const uint32_t *vertex)
{
    // assert(false);
}

/**
 * @note `buffer_size`实际上是第二大节点度数，而非所用空间大小
 * @todo 接口类型可能需要修改
 */
__global__ void gpu_pattern_matching(uint32_t edge_num, uint32_t buffer_size,
    const uint32_t *edge_from, uint32_t *edge, const uint32_t *vertex, uint32_t *tmp, const GPUSchedule* schedule) {
    __shared__ unsigned int block_edge_idx[WARPS_PER_BLOCK];
    //之后考虑把tmp buffer都放到shared里来（如果放得下）
    extern __shared__ uint8_t block_shmem[];
    
    const int num_prefixes = schedule->get_total_prefix_num();
    const int num_vertex_sets_per_warp = num_prefixes + 1;

    int wid = threadIdx.x / THREADS_PER_WARP; // warp id within the block
    int lid = threadIdx.x % THREADS_PER_WARP; // lane id
    int global_wid = blockIdx.x * WARPS_PER_BLOCK + wid; // global warp id
    
    unsigned int &edge_idx = block_edge_idx[wid];
    const size_t warp_shmem_size = sizeof(GPUSubtractionSet) + num_vertex_sets_per_warp * sizeof(GPUVertexSet);   
    uint8_t *warp_shmem = block_shmem + wid * warp_shmem_size;
    GPUSubtractionSet &subtraction_set = *reinterpret_cast<GPUSubtractionSet*>(warp_shmem);
    GPUVertexSet *vertex_set = reinterpret_cast<GPUVertexSet*>(warp_shmem + sizeof(GPUSubtractionSet));
    GPUVertexSet &tmp_set = vertex_set[num_prefixes];

    if (lid == 0) {
        edge_idx = 0;
        ptrdiff_t offset = ptrdiff_t(1) * buffer_size * global_wid * num_vertex_sets_per_warp;
        for (int i = 0; i < num_vertex_sets_per_warp; ++i)
        {
            vertex_set[i].set_data_ptr(tmp + offset); // 注意这是个指针+整数运算，自带*4
            offset += buffer_size;
        }
    }

    __threadfence_block(); //之后考虑把所有的syncthreads都改成syncwarp

    uint32_t v0, v1;
    uint32_t l, r;

    unsigned long long local_sum, sum = 0;

    while (true) {
        if (lid == 0) {
            //if(++edgeI >= edgeEnd) { //这个if语句应该是每次都会发生吧？（是的
                edge_idx = atomicAdd(&dev_cur_edge, 1);
                //edgeEnd = min(edge_num, edgeI + 1); //这里不需要原子读吗
                unsigned int i = edge_idx;
                if (i < edge_num)
                {
                    subtraction_set.init();
                    subtraction_set.push_back(edge_from[i]);
                    subtraction_set.push_back(edge[i]);
                }
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
        for (int prefix_id = schedule->get_last(0); prefix_id != -1; prefix_id = schedule->get_next(prefix_id))
            vertex_set[prefix_id].build_vertex_set(schedule, vertex_set, &edge[l], r - l, prefix_id);

        //目前只考虑pattern size>2的情况
        //start v1, depth = 1
        if (schedule->get_restrict_last(1) != -1 && v0 <= v1)
            continue;
        
        get_edge_index(v1, l, r);
        for (int prefix_id = schedule->get_last(1); prefix_id != -1; prefix_id = schedule->get_next(prefix_id))
        {
            vertex_set[prefix_id].build_vertex_set(schedule, vertex_set, &edge[l], r - l, prefix_id);
            if (vertex_set[prefix_id].get_size() == 0) {
                is_zero = true;
                break;
            }
        }
        if (is_zero)
            continue;
        
        local_sum = 0; // local sum (corresponding to an edge index)
        GPU_pattern_matching_func<2>(schedule, vertex_set, subtraction_set, tmp_set, local_sum, edge, vertex);
        // GPU_pattern_matching_aggressive_func(schedule, vertex_set, subtraction_set, tmp_set, local_sum, 2, edge, vertex);
        sum += local_sum;
    }

    if (lid == 0) {
        atomicAdd(&dev_sum, sum);
    }
}

template <typename T, std::size_t S>
T sum_of_dev_array(T (&arr)[S])
{
    T sum = 0;
    T *tmp = new T[S];
    gpuErrchk( cudaMemcpyFromSymbol(tmp, arr, sizeof(arr)) );
    for (std::size_t i = 0; i < S; ++i)
        sum += tmp[i];
    delete[] tmp;
    return sum;
}

/**
 * @return counting time in seconds
 */ 
double pattern_matching_entry(const Graph *g, const Schedule& schedule)
{
    if (schedule.get_size() - schedule.get_in_exclusion_optimize_num() > GPUSubtractionSet::MAX_NR_ELEMENTS) {
        printf("GPUSubtractionSet is not big enough to hold all elements. Please recompile and run the program.\n");
        return 0;
    }

    int device;
    gpuErrchk( cudaGetDevice(&device) );

    int num_sms; // number of Streaming Multiprocessors
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device);

    size_t buffer_size = VertexSet::max_intersection_size; // TODO: move field VertexSet::max_intersection_size into Graph
    size_t warp_shmem_size = sizeof(GPUSubtractionSet) + (schedule.get_total_prefix_num() + 1) * sizeof(GPUVertexSet);
    size_t block_shmem_size = warp_shmem_size * WARPS_PER_BLOCK;

    printf("schedule.prefix_num: %d\n", schedule.get_total_prefix_num());
    printf("shared memory for vertex set per block: %lu bytes\n", block_shmem_size);

    int max_active_blocks_per_sm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks_per_sm, gpu_pattern_matching, THREADS_PER_BLOCK, block_shmem_size);
    printf("max number of active warps per SM: %d\n", max_active_blocks_per_sm * WARPS_PER_BLOCK);
    
    int num_blocks = max_active_blocks_per_sm * num_sms;
    int num_total_warps = num_blocks * WARPS_PER_BLOCK;
    printf("number of blocks: %d number of total warps: %d\n", num_blocks, num_total_warps);

    size_t size_edge = g->e_cnt * sizeof(uint32_t);
    size_t size_vertex = (g->v_cnt + 1) * sizeof(uint32_t);
    size_t size_tmp = buffer_size * sizeof(uint32_t) * num_total_warps * (schedule.get_total_prefix_num() + 1); //prefix + tmp

    schedule.print_schedule();
    printf("Restrictions:");
    for (const auto& pair : schedule.restrict_pair)
            printf(" (%d,%d)", pair.first, pair.second);
    printf("\n");

    uint32_t *edge_from = new uint32_t[g->e_cnt];
    for(uint32_t i = 0; i < g->v_cnt; ++i)
        for(uint32_t j = g->vertex[i]; j < g->vertex[i+1]; ++j)
            edge_from[j] = i;

    using std::chrono::system_clock;
    auto t1 = system_clock::now();

    uint32_t *dev_edge;
    uint32_t *dev_edge_from;
    uint32_t *dev_vertex;
    uint32_t *dev_tmp;

    gpuErrchk( cudaMalloc((void**)&dev_edge, size_edge));
    gpuErrchk( cudaMalloc((void**)&dev_edge_from, size_edge));
    gpuErrchk( cudaMalloc((void**)&dev_vertex, size_vertex));
    gpuErrchk( cudaMalloc((void**)&dev_tmp, size_tmp));

    gpuErrchk( cudaMemcpy(dev_edge, g->edge, size_edge, cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy(dev_edge_from, edge_from, size_edge, cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy(dev_vertex, g->vertex, size_vertex, cudaMemcpyHostToDevice));

    unsigned long long sum = 0;
    unsigned int initial_edge_index = 0;

    // initialize global device variables
    gpuErrchk( cudaMemcpyToSymbol(dev_cur_edge, &initial_edge_index, sizeof(dev_cur_edge)) );
    gpuErrchk( cudaMemcpyToSymbol(dev_sum, &sum, sizeof(dev_sum)) );

    //memcpy schedule
    GPUSchedule* dev_schedule;
    gpuErrchk( cudaMallocManaged(&dev_schedule, sizeof(GPUSchedule)));
    dev_schedule->init_from(schedule);

    auto t2 = system_clock::now();
    auto prepare_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    printf("Prepare time cost: %g seconds\n", 1e-6 * prepare_time.count());
    fflush(stdout);

    auto t3 = system_clock::now();

    //因为目前用了managed开内存，所以第一次运行kernel会有一定额外开销，考虑运行两次，第一次作为warmup
    gpu_pattern_matching<<<num_blocks, THREADS_PER_BLOCK, block_shmem_size>>>
        (g->e_cnt, buffer_size, dev_edge_from, dev_edge, dev_vertex, dev_tmp, dev_schedule);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk( cudaMemcpyFromSymbol(&sum, dev_sum, sizeof(sum)) );

    auto t4 = system_clock::now();
    double counting_time = 1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();

    printf("count %llu\n", sum);
    printf("Counting time cost: %g seconds\n", counting_time);
    fflush(stdout);

    // 释放内存
    gpuErrchk(cudaFree(dev_edge));
    gpuErrchk(cudaFree(dev_edge_from));
    gpuErrchk(cudaFree(dev_vertex));
    gpuErrchk(cudaFree(dev_tmp));

    dev_schedule->release();
    gpuErrchk(cudaFree(dev_schedule));

    delete[] edge_from;

    return counting_time;
}

static inline bool is_adj_matrix(const char* s, int& pattern_size)
{
    int n, i;
    for (i = 0; s[i] == '0' || s[i] == '1'; ++i)
        ;
    if (i <= 1)
        return false;
    n = static_cast<int>(std::sqrt(i));
    if (n * n != i)
        return false;
    pattern_size = n;
    return true;
}

/**
 * @return minimum counting time of all schedules
 */
double test_all_schedules(Graph *g, int pattern_size, const char* adj_mat)
{
    int size = pattern_size;
    int rank[size], perm_id = 0;
    for (int i = 0; i < size; ++i)
        rank[i] = i;

    double min_counting_time = std::numeric_limits<double>::max();
    
    do {
        printf("\n-------------- permutation id = %d ---------------\n", ++perm_id);

        Pattern cur_pattern(size);
        for (int i = 0; i < size; ++i)
            for (int j = 0; j < size; ++j)
                if (adj_mat[INDEX(i, j, size)] == '1')
                    cur_pattern.add_edge(rank[i], rank[j]);
        
        bool valid;
        Schedule *schedule_ptr = nullptr;
        try {
            schedule_ptr = new Schedule(cur_pattern, valid, 0, 1, true, g->v_cnt, g->e_cnt, g->tri_cnt);
        } catch (const std::exception& e) {
            valid = false;
        }
        if (!valid || !schedule_ptr) {
            printf("invalid schedule.\n");
            continue;
        }

        if (schedule_ptr->get_in_exclusion_optimize_num() == 0) {
            printf("in-exclusion optimization is not available.\n");
            continue;
        }

        double counting_time = pattern_matching_entry(g, *schedule_ptr);
        if (min_counting_time > counting_time)
            min_counting_time = counting_time;

        printf("----------------------------------------------------\n");
    } while (std::next_permutation(rank, rank + size));

    return min_counting_time;
}

int main(int argc,char *argv[]) {
    Graph *g;
    DataLoader D;

    if (argc < 2) {
        printf("Usage: %s dataset_name graph_file\n", argv[0]);
        printf("Example: %s Patents ~zms/patents_input\n", argv[0]);

        printf("\nExperimental usage: %s [graph_file.g] [adjacency_matrix]\n", argv[0]);
        printf("Example: %s ~hzx/data/patents.g\n", argv[0]);
        printf("Example: %s ~hzx/data/mico.g 0111010011100011100001100", argv[0]);
        return 0;
    }

    // default pattern: house p1
    int pattern_size = 5;
    const char *adj_mat = "0111010011100011100001100"; // 5 house p1
    // const char *adj_mat = "011011101110110101011000110000101000"; // 6 p2
    // const char *adj_mat = "011110101101110000110000100001010010"; // 6 p4
    // const char *adj_mat = "0111111101111111011101110100111100011100001100000"; // 7 p5
    // const char *adj_mat = "0111111101111111011001110100111100011000001100000"; // 7 p6

    bool binary_input = true;
    DataType my_type;

    if (argc >= 3) {
        if (is_adj_matrix(argv[2], pattern_size)) {
            adj_mat = argv[2];
        } else {
            printf("'%s' is not an adjacency matrix. Assume old-style input.\n", argv[2]);

            binary_input = false;
            GetDataType(my_type, argv[1]);

            if (my_type == DataType::Invalid) {
                printf("Dataset not found!\n");
                return 0;
            }
        }
    }

    using std::chrono::system_clock;
    auto t1 = system_clock::now();

    bool ok;
    if (!binary_input) {
        // 注：load_data的第四个参数用于指定是否读取【旧式】二进制文件输入，默认为false
        // 另注：【旧式】二进制输入已不再使用
        ok = D.load_data(g, my_type, argv[2], false);
    } else {
        ok = D.fast_load(g, argv[1]); // 快速读取预处理过的.g图文件
    }
    if (!ok) {
        printf("data load failure :-(\n");
        return 0;
    }

    auto t2 = system_clock::now();
    auto load_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    printf("Load data success! time: %g seconds\n", load_time.count() / 1.0e6);
    fflush(stdout);

    allTime.check();

    Pattern p(pattern_size, adj_mat);
    printf("pattern(%d)=\n", pattern_size);
    p.print();
    printf("max intersection size %d\n", VertexSet::max_intersection_size);
    fflush(stdout);

    bool is_pattern_valid;
    bool use_in_exclusion_optimize = true;

    // use the best schedule
    Schedule schedule(p, is_pattern_valid, 1, 1, use_in_exclusion_optimize, g->v_cnt, g->e_cnt, g->tri_cnt);
    //Schedule schedule(p, is_pattern_valid, 0, 1, use_in_exclusion_optimize, g->v_cnt, g->e_cnt, g->tri_cnt); // use the best schedule
    if (!is_pattern_valid) {
        printf("pattern is invalid!\n");
        return 0;
    }

    
    double default_counting_time = pattern_matching_entry(g, schedule);

    /*
    double min_counting_time = test_all_schedules(g, pattern_size, adj_mat);
    printf("\n\n------------- summary --------------\n\n");
    if (min_counting_time < default_counting_time) {
        printf("current schedule: %g seconds; best: %g seconds\n", default_counting_time, min_counting_time);
        printf("current schedule => %g%% slower.\n", 100.0 * (default_counting_time - min_counting_time) / min_counting_time);
    } else {
        printf("current schedule is the best.\n");
    }
    */

    allTime.print("Total time cost");

    return 0;
}
