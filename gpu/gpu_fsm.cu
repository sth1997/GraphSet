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

constexpr int THREADS_PER_BLOCK = 128;
//constexpr int THREADS_PER_BLOCK = 32;
constexpr int THREADS_PER_WARP = 32;
constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / THREADS_PER_WARP;

//constexpr int num_blocks = 1024;
constexpr int num_blocks = 1024;
constexpr int num_total_warps = num_blocks * WARPS_PER_BLOCK;

__device__ unsigned long long dev_sum = 0;
__device__ unsigned int dev_cur_labeled_pattern = 0;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define get_labeled_edge_index(v, label, l, r) do { \
    int index = v * l_cnt + label; \
    l = labeled_vertex[index]; \
    r = labeled_vertex[index+ 1]; \
} while(0)

template <typename T>
__device__ inline void swap(T& a, T& b)
{
    T t(std::move(a));
    a = std::move(b);
    b = std::move(t);
}

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

class GPUSchedule {
public:
    inline __device__ int get_total_prefix_num() const { return total_prefix_num;}
    //inline __device__ int get_basic_prefix_num() const { return basic_prefix_num;}
    inline __device__ int get_father_prefix_id(int prefix_id) const { return father_prefix_id[prefix_id];}
    inline __device__ int get_loop_set_prefix_id(int loop) const { return loop_set_prefix_id[loop];}
    inline __device__ int get_size() const { return size;}
    inline __device__ int get_last(int i) const { return last[i];}
    inline __device__ int get_next(int i) const { return next[i];}
    inline __device__ int get_prefix_target(int i) const {return prefix_target[i];}
    //inline __device__ int get_break_size(int i) const { return break_size[i];}
    //inline __device__ int get_in_exclusion_optimize_num() const { return in_exclusion_optimize_num;}
    // inline __device__ int get_total_restrict_num() const { return total_restrict_num;}
    // inline __device__ int get_restrict_last(int i) const { return restrict_last[i];}
    // inline __device__ int get_restrict_next(int i) const { return restrict_next[i];}
    // inline __device__ int get_restrict_index(int i) const { return restrict_index[i];}
    //inline __device__ int get_k_val() const { return k_val;} // see below (the k_val's definition line) before using this function

    //int* adj_mat;
    int* father_prefix_id;
    int* last;
    int* next;
    //int* break_size;
    int* loop_set_prefix_id;
    int* prefix_target;
    // int* restrict_last;
    // int* restrict_next;
    // int* restrict_index;
    //bool* only_need_size;
    //int* in_exclusion_optimize_val;
    //GPUGroupDim0 in_exclusion_optimize_group;
    //int in_exclusion_optimize_val_size;
    int size;
    int total_prefix_num;
    //int basic_prefix_num;
    //int total_restrict_num;
    //int in_exclusion_optimize_num;
    //int k_val;

    // int in_exclusion_optimize_vertex_id_size;
    // int* in_exclusion_optimize_vertex_id;
    // bool* in_exclusion_optimize_vertex_flag;
    // int* in_exclusion_optimize_vertex_coef;
    
    // int in_exclusion_optimize_array_size;
    // int* in_exclusion_optimize_coef;
    // bool* in_exclusion_optimize_flag;
    // int* in_exclusion_optimize_ans_pos;

    //uint32_t ans_array_offset;
    uint32_t p_label_offset;
    int max_edge;
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
        // for (int i = lid; i < input_size; i += THREADS_PER_WARP){
        //     data[i] = input_data[i];
        // }
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

class GPUBitVector {
public:
    void construct(size_t element_cnt) {
        size = (element_cnt + 31) / 32;
        gpuErrchk( cudaMalloc((void**)&data, size * sizeof(uint32_t)));
    }
    void destroy() {
        gpuErrchk(cudaFree(data));
    }
    __device__ void clear() {
        non_zero_cnt = 0;
        memset((void*) data, 0, size * sizeof(uint32_t));
    }
    GPUBitVector& operator = (const GPUBitVector&) = delete;
    GPUBitVector(const GPUBitVector&&) = delete;
    GPUBitVector(const GPUBitVector&) = delete;
    inline __device__ long long get_non_zero_cnt() const { return non_zero_cnt;}
    __device__ void insert(uint32_t id) { //目前每个warp一个bitvector，且只有一个线程会操作，所以不加锁
        uint32_t index = id >> 5;
        uint32_t tmp_data = data[index];
        uint32_t offset = 1 << (id % 32);
        if ((tmp_data & offset) == 0) {
            ++non_zero_cnt;
            data[index] = tmp_data | offset;
        }
    }
private:
    long long non_zero_cnt;
    size_t size;
    uint32_t* data;
};

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
     __shared__ uint32_t block_out_offset[THREADS_PER_BLOCK];
     __shared__ uint32_t block_out_size[WARPS_PER_BLOCK];
 
     int wid = threadIdx.x / THREADS_PER_WARP; // warp id
     int lid = threadIdx.x % THREADS_PER_WARP; // lane id
     uint32_t *out_offset = block_out_offset + wid * THREADS_PER_WARP;
     uint32_t &out_size = block_out_size[wid];
 
     if (lid == 0)
         out_size = 0;
 
     for(int num_done = 0; num_done < na; num_done += THREADS_PER_WARP) {
         bool found = 0;
         uint32_t u = 0;
         if (num_done + lid < na) {
             u = a[num_done + lid]; // u: an element in set a
             /*
             // 我这里这样写并没有变快，反而明显慢了
             int x, s[3], &l = s[0], &r = s[1], &mid = s[2];
             l = 0, r = int(nb) - 1, mid = (int(nb) - 1) >> 1;
             while (l <= r && !found) {
                 uint32_t v = b[mid];
                 found = (v == u);
                 x = (v < u);
                 mid += 2 * x - 1;
                 swap(mid, s[!x]);
                 mid = (l + r) >> 1;
             }
             */
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
         out_offset[lid] = found;
         __threadfence_block();
 
         #pragma unroll
         for (int s = 1; s < THREADS_PER_WARP; s *= 2) {
             uint32_t v = lid >= s ? out_offset[lid - s] : 0;
             __threadfence_block();
             out_offset[lid] += v;
             __threadfence_block();
         }
         
         /*
         // work-efficient parallel scan，但常数大，实测速度不行
         #pragma unroll
         for (int s = 1; s < THREADS_PER_WARP; s <<= 1) {
             int i = (lid + 1) * s * 2 - 1;
             if (i < THREADS_PER_WARP)
                 out_offset[i] += out_offset[i - s];
             __threadfence_block();
         }
 
         #pragma unroll
         for (int s = THREADS_PER_WARP / 4; s > 0; s >>= 1) {
             int i = (lid + 1) * s * 2 - 1;
             if ((i + s) < THREADS_PER_WARP)
                 out_offset[i + s] += out_offset[i];
             __threadfence_block();
         }
         */
         
         if (found) {
             uint32_t offset = out_offset[lid] - 1;
             out[out_size + offset] = u;
         }
 
         if (lid == 0)
             out_size += out_offset[THREADS_PER_WARP - 1];
 
         /*
         // 使用warp shuffle的scan，但实测速度更不行
         uint32_t offset = found;
         #pragma unroll
         for (int i = 1; i < THREADS_PER_WARP; i *= 2) {
             uint32_t t = __shfl_up_sync(0xffffffff, offset, i);
             if (lid >= i)
                 offset += t;
         }
 
         if (found)
             out[out_size + offset - 1] = u;
         if (lid == THREADS_PER_WARP - 1) // 总和被warp中最后一个线程持有
             out_size += offset;
         */
 
         //num_done += THREADS_PER_WARP;
     }
 
     __threadfence_block();
     return out_size;
 }

/**
 * wrapper of search based intersection `do_intersection`
 * 
 * 注意：不能进行in-place操作。若想原地操作则应当把交换去掉。
 */
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

constexpr int MAX_DEPTH = 5; // 非递归pattern matching支持的最大深度

template <int depth>
__device__ bool GPU_pattern_matching_func(const GPUSchedule* schedule, GPUVertexSet* vertex_set, GPUVertexSet& subtraction_set,
    uint32_t *edge, uint32_t* labeled_vertex, const char* p_label, GPUBitVector* fsm_set, int l_cnt)
{
    // if(threadIdx.x % THREADS_PER_WARP == 0 && depth == 3)
    //     printf("%d",depth);
    int loop_set_prefix_id = schedule->get_loop_set_prefix_id(depth);
    int loop_size = vertex_set[loop_set_prefix_id].get_size();
    if (loop_size <= 0) //这个判断可能可以删了
        return false;
    uint32_t* loop_data_ptr = vertex_set[loop_set_prefix_id].get_data_ptr();

    bool local_match = false;
    __shared__ bool block_match[WARPS_PER_BLOCK];
    if (depth == schedule->get_size() - 1) {
        if (threadIdx.x % THREADS_PER_WARP == 0) { //TODO: 改成并行，不过要注意现在fsm_set不支持并行
            for (int i = 0; i < loop_size; ++i)
            {
                int vertex = loop_data_ptr[i];
                if (subtraction_set.has_data(vertex))
                    continue;
                local_match = true;
                fsm_set[depth].insert(vertex);
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
        if (subtraction_set.has_data(v))
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
            subtraction_set.push_back(v);
        __threadfence_block();

        if (GPU_pattern_matching_func<depth + 1>(schedule, vertex_set, subtraction_set, edge, labeled_vertex, p_label, fsm_set, l_cnt)) {
            local_match = true;
            if (threadIdx.x % THREADS_PER_WARP == 0)
                fsm_set[depth].insert(v);
        }
        if (threadIdx.x % THREADS_PER_WARP == 0)
            subtraction_set.pop_back();
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
 __global__ void gpu_pattern_matching(uint32_t job_num, uint32_t v_cnt, uint32_t buffer_size, uint32_t *edge, uint32_t* labeled_vertex, int* v_label, uint32_t* tmp, const GPUSchedule* schedule, char* all_p_label, GPUBitVector* global_fsm_set, int* automorphisms, unsigned int* is_frequent, unsigned int* label_start_idx, int automorphisms_cnt, long long min_support, unsigned int pattern_is_frequent_index, int l_cnt) {
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


    //uint32_t v0, v1;
    //uint32_t l, r;

    //unsigned long long sum = 0;

    while (true) {
        if (lid == 0) {
            //if(++edgeI >= edgeEnd) { //这个if语句应该是每次都会发生吧？（是的
                pattern_idx = atomicAdd(&dev_cur_labeled_pattern, 1); //每个warp负责一个pattern，而不是负责一个点或一条边
                //edgeEnd = min(edge_num, edgeI + 1); //这里不需要原子读吗
                unsigned int job_id = pattern_idx;
                if (job_id < job_num)
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
        if(job_id >= job_num) break;

        if (lid < schedule->get_size())
            fsm_set[lid].clear();
        __threadfence_block();
        
        //for (int vertex = 0; vertex < v_cnt; ++vertex)
        //    if (v_label[vertex] == p_label[0]) {//TODO: 这里也可以换成一个提前按照v_label排序，会快一些
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
                    subtraction_set.push_back(vertex);
                __threadfence_block();
                // if(lid == 0) {
                //     printf("%d\n", vertex);
                // }
                if (GPU_pattern_matching_func<1>(schedule, vertex_set, subtraction_set, edge, labeled_vertex, p_label, fsm_set, l_cnt))
                    if (lid == 0) //TODO: 目前insert只让0号线程执行，之后考虑32个线程同时执行，看会不会出错（好像是不会）
                    {
                        fsm_set[0].insert(vertex);
                    }
                if (lid == 0)
                    subtraction_set.pop_back();
                __threadfence_block();
                if (block_break_flag[wid] == true)
                    break;
        }
        if(lid == 0) {
            long long support = v_cnt;
            for (int i = 0; i < schedule->get_size(); ++i) {
                long long count = fsm_set[i].get_non_zero_cnt();
                if (count < support)
                    support = count;
            }
            // printf("%d\n", support);
            if (support >= min_support) {
                printf("support: %lld job_id:%d\n", support, job_id);
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
            }
        }

        /*if (lid == 0) {
            long long support = v_cnt;
            for (int i = 0; i < schedule->get_size(); ++i) {
                long long count = fsm_set[i].get_non_zero_cnt();
                if (count < support)
                    support = count;
            }
            if (support >= min_support) {
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
            }
        }*/
        __threadfence_block();
        /*
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
            if (vertex_set[prefix_id].get_size() == 0 && prefix_id < schedule->get_basic_prefix_num()) {
                is_zero = true;
                break;
            }
        }
        if (is_zero)
            continue;
        
        //unsigned long long local_sum = 0; // local sum (corresponding to an edge index)
        */
        //GPU_pattern_matching_func<2>(schedule, vertex_set, subtraction_set, tmp_set, edge, vertex);
        // GPU_pattern_matching_aggressive_func(schedule, vertex_set, subtraction_set, tmp_set, local_sum, 2, edge, vertex);
        //sum += local_sum;
    }
}

long long pattern_matching_init(const LabeledGraph *g, const Schedule_IEP& schedule, const std::vector<std::vector<int> >& automorphisms, unsigned int pattern_is_frequent_index, unsigned int* dev_is_frequent, uint32_t* dev_edge, uint32_t* dev_labeled_vertex, int* dev_v_label, uint32_t* dev_tmp, int max_edge, int job_num, char* dev_all_p_label, GPUBitVector* fsm_set, uint32_t* dev_label_start_idx, long long min_support) {
    //printf("basic prefix %d, total prefix %d\n", schedule.get_basic_prefix_num(), schedule.get_total_prefix_num());
    printf("total prefix %d\n", schedule.get_total_prefix_num());

    schedule.print_schedule();
    //uint32_t *edge_from = new uint32_t[g->e_cnt];
    //for(uint32_t i = 0; i < g->v_cnt; ++i)
    //    for(uint32_t j = g->vertex[i]; j < g->vertex[i+1]; ++j)
    //        edge_from[j] = i;

    tmpTime.check(); 

    unsigned long long sum = 0; //sum是这个pattern的所有labeled pattern中频繁的个数
    gpuErrchk( cudaMemcpyToSymbol(dev_sum, &sum, sizeof(sum)));
    unsigned int cur_labeled_pattern = 0;
    gpuErrchk( cudaMemcpyToSymbol(dev_cur_labeled_pattern, &cur_labeled_pattern, sizeof(cur_labeled_pattern)));
    
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

    //memcpy schedule
    GPUSchedule* dev_schedule;
    gpuErrchk( cudaMallocManaged((void**)&dev_schedule, sizeof(GPUSchedule)));
    //dev_schedule->transform_in_exclusion_optimize_group_val(schedule);
    int schedule_size = schedule.get_size();
    int max_prefix_num = schedule_size * (schedule_size - 1) / 2;
    
    //bool *only_need_size = new bool[max_prefix_num];
    //for(int i = 0; i < max_prefix_num; ++i)
    //    only_need_size[i] = schedule.get_prefix_only_need_size(i);

    //int in_exclusion_optimize_vertex_id_size = schedule.in_exclusion_optimize_vertex_id.size();
    //int in_exclusion_optimize_array_size  = schedule.in_exclusion_optimize_coef.size();

    //assert(in_exclusion_optimize_array_size == schedule.in_exclusion_optimize_coef.size());
    //assert(in_exclusion_optimize_array_size == schedule.in_exclusion_optimize_flag.size());

    //printf("array size %d\n", in_exclusion_optimize_array_size);
    //fflush(stdout);

    // int* in_exclusion_optimize_vertex_id = new int[in_exclusion_optimize_vertex_id_size];
    // bool* in_exclusion_optimize_vertex_flag = new bool[in_exclusion_optimize_vertex_id_size];
    // int* in_exclusion_optimize_vertex_coef = new int[in_exclusion_optimize_vertex_id_size];
    
    // int* in_exclusion_optimize_coef = new int[in_exclusion_optimize_array_size];
    // bool* in_exclusion_optimize_flag = new bool[in_exclusion_optimize_array_size];
    // int* in_exclusion_optimize_ans_pos = new int[in_exclusion_optimize_array_size];

    // for(int i = 0; i < in_exclusion_optimize_vertex_id_size; ++i) {
    //     in_exclusion_optimize_vertex_id[i] = schedule.in_exclusion_optimize_vertex_id[i];
    //     in_exclusion_optimize_vertex_flag[i] = schedule.in_exclusion_optimize_vertex_flag[i];
    //     in_exclusion_optimize_vertex_coef[i] = schedule.in_exclusion_optimize_vertex_coef[i];
    // }

    // for(int i = 0; i < in_exclusion_optimize_array_size; ++i) {
    //     in_exclusion_optimize_coef[i] = schedule.in_exclusion_optimize_coef[i];
    //     in_exclusion_optimize_flag[i] = schedule.in_exclusion_optimize_flag[i];
    //     in_exclusion_optimize_ans_pos[i] = schedule.in_exclusion_optimize_ans_pos[i];
    // }

    // if (in_exclusion_optimize_vertex_id_size > 0) {
    //     gpuErrchk( cudaMallocManaged((void**)&dev_schedule->in_exclusion_optimize_vertex_id, sizeof(int) * in_exclusion_optimize_vertex_id_size));
    //     gpuErrchk( cudaMemcpy(dev_schedule->in_exclusion_optimize_vertex_id, in_exclusion_optimize_vertex_id, sizeof(int) * in_exclusion_optimize_vertex_id_size, cudaMemcpyHostToDevice));
        
    //     gpuErrchk( cudaMallocManaged((void**)&dev_schedule->in_exclusion_optimize_vertex_flag, sizeof(bool) * in_exclusion_optimize_vertex_id_size));
    //     gpuErrchk( cudaMemcpy(dev_schedule->in_exclusion_optimize_vertex_flag, in_exclusion_optimize_vertex_flag, sizeof(bool) * in_exclusion_optimize_vertex_id_size, cudaMemcpyHostToDevice));
        
    //     gpuErrchk( cudaMallocManaged((void**)&dev_schedule->in_exclusion_optimize_vertex_coef, sizeof(int) * in_exclusion_optimize_vertex_id_size));
    //     gpuErrchk( cudaMemcpy(dev_schedule->in_exclusion_optimize_vertex_coef, in_exclusion_optimize_vertex_coef, sizeof(int) * in_exclusion_optimize_vertex_id_size, cudaMemcpyHostToDevice));
    // }

    // if (in_exclusion_optimize_array_size > 0)
    // {
    //     gpuErrchk( cudaMallocManaged((void**)&dev_schedule->in_exclusion_optimize_coef, sizeof(int) * in_exclusion_optimize_array_size));
    //     gpuErrchk( cudaMemcpy(dev_schedule->in_exclusion_optimize_coef, in_exclusion_optimize_coef, sizeof(int) * in_exclusion_optimize_array_size, cudaMemcpyHostToDevice));

    //     gpuErrchk( cudaMallocManaged((void**)&dev_schedule->in_exclusion_optimize_flag, sizeof(bool) * in_exclusion_optimize_array_size));
    //     gpuErrchk( cudaMemcpy(dev_schedule->in_exclusion_optimize_flag, in_exclusion_optimize_flag, sizeof(bool) * in_exclusion_optimize_array_size, cudaMemcpyHostToDevice));
        
    //     gpuErrchk( cudaMallocManaged((void**)&dev_schedule->in_exclusion_optimize_ans_pos, sizeof(int) * in_exclusion_optimize_array_size));
    //     gpuErrchk( cudaMemcpy(dev_schedule->in_exclusion_optimize_ans_pos, in_exclusion_optimize_ans_pos, sizeof(int) * in_exclusion_optimize_array_size, cudaMemcpyHostToDevice));
    // }

    //gpuErrchk( cudaMallocManaged((void**)&dev_schedule->adj_mat, sizeof(int) * schedule_size * schedule_size));
    //gpuErrchk( cudaMemcpy(dev_schedule->adj_mat, schedule.get_adj_mat_ptr(), sizeof(int) * schedule_size * schedule_size, cudaMemcpyHostToDevice));

    gpuErrchk( cudaMallocManaged((void**)&dev_schedule->father_prefix_id, sizeof(int) * max_prefix_num));
    gpuErrchk( cudaMemcpy(dev_schedule->father_prefix_id, schedule.get_father_prefix_id_ptr(), sizeof(int) * max_prefix_num, cudaMemcpyHostToDevice));

    gpuErrchk( cudaMallocManaged((void**)&dev_schedule->last, sizeof(int) * schedule_size));
    gpuErrchk( cudaMemcpy(dev_schedule->last, schedule.get_last_ptr(), sizeof(int) * schedule_size, cudaMemcpyHostToDevice));

    gpuErrchk( cudaMallocManaged((void**)&dev_schedule->next, sizeof(int) * max_prefix_num));
    gpuErrchk( cudaMemcpy(dev_schedule->next, schedule.get_next_ptr(), sizeof(int) * max_prefix_num, cudaMemcpyHostToDevice));
    
    //gpuErrchk( cudaMallocManaged((void**)&dev_schedule->only_need_size, sizeof(bool) * max_prefix_num));
    //gpuErrchk( cudaMemcpy(dev_schedule->only_need_size, only_need_size, sizeof(bool) * max_prefix_num, cudaMemcpyHostToDevice));
    //TODO: 使用break size
    //gpuErrchk( cudaMallocManaged((void**)&dev_schedule->break_size, sizeof(int) * max_prefix_num));
    //gpuErrchk( cudaMemcpy(dev_schedule->break_size, schedule.get_break_size_ptr(), sizeof(int) * max_prefix_num, cudaMemcpyHostToDevice));

    gpuErrchk( cudaMallocManaged((void**)&dev_schedule->loop_set_prefix_id, sizeof(int) * schedule_size));
    gpuErrchk( cudaMemcpy(dev_schedule->loop_set_prefix_id, schedule.get_loop_set_prefix_id_ptr(), sizeof(int) * schedule_size, cudaMemcpyHostToDevice));

    gpuErrchk( cudaMallocManaged((void**)&dev_schedule->prefix_target, sizeof(int) * max_prefix_num));
    gpuErrchk( cudaMemcpy(dev_schedule->prefix_target, schedule.get_prefix_target_ptr(), sizeof(int) * max_prefix_num, cudaMemcpyHostToDevice));

    // gpuErrchk( cudaMallocManaged((void**)&dev_schedule->restrict_last, sizeof(int) * schedule_size));
    // gpuErrchk( cudaMemcpy(dev_schedule->restrict_last, schedule.get_restrict_last_ptr(), sizeof(int) * schedule_size, cudaMemcpyHostToDevice));
    
    // gpuErrchk( cudaMallocManaged((void**)&dev_schedule->restrict_next, sizeof(int) * max_prefix_num));
    // gpuErrchk( cudaMemcpy(dev_schedule->restrict_next, schedule.get_restrict_next_ptr(), sizeof(int) * max_prefix_num, cudaMemcpyHostToDevice));
    
    // gpuErrchk( cudaMallocManaged((void**)&dev_schedule->restrict_index, sizeof(int) * max_prefix_num));
    // gpuErrchk( cudaMemcpy(dev_schedule->restrict_index, schedule.get_restrict_index_ptr(), sizeof(int) * max_prefix_num, cudaMemcpyHostToDevice));

    //dev_schedule->in_exclusion_optimize_array_size = in_exclusion_optimize_array_size;
    //dev_schedule->in_exclusion_optimize_vertex_id_size = in_exclusion_optimize_vertex_id_size;
    dev_schedule->size = schedule.get_size();
    dev_schedule->total_prefix_num = schedule.get_total_prefix_num();
    //dev_schedule->basic_prefix_num = schedule.get_basic_prefix_num();
    //dev_schedule->total_restrict_num = schedule.get_total_restrict_num();
    //dev_schedule->in_exclusion_optimize_num = schedule.get_in_exclusion_optimize_num();
    //dev_schedule->k_val = schedule.get_k_val();
    
    printf("schedule.prefix_num: %d\n", schedule.get_total_prefix_num());
    //printf("shared memory for vertex set per block: %ld bytes\n", 
    //    (schedule.get_total_prefix_num() + 2) * WARPS_PER_BLOCK * sizeof(GPUVertexSet) + in_exclusion_optimize_vertex_id_size * WARPS_PER_BLOCK * sizeof(int));
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
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks_per_sm, gpu_pattern_matching, THREADS_PER_BLOCK, block_shmem_size);
    printf("max number of active warps per SM: %d\n", max_active_blocks_per_sm * WARPS_PER_BLOCK);
    
    //gpu_pattern_matching<<<num_blocks, THREADS_PER_BLOCK, block_shmem_size>>>
    //    (g->e_cnt, buffer_size, dev_edge_from, dev_edge, dev_vertex, dev_tmp, dev_schedule);
    gpu_pattern_matching<<<num_blocks, THREADS_PER_BLOCK, block_shmem_size>>>
        (job_num, g->v_cnt, buffer_size, dev_edge, dev_labeled_vertex, dev_v_label, dev_tmp, dev_schedule, dev_all_p_label, fsm_set, dev_automorphisms, dev_is_frequent, dev_label_start_idx, automorphisms.size(), min_support, pattern_is_frequent_index, g->l_cnt);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk( cudaMemcpyFromSymbol(&sum, dev_sum, sizeof(sum)) );

    //sum /= schedule.get_in_exclusion_optimize_redundancy();
    
    #ifdef PRINT_ANS_TO_FILE
    freopen("1.out", "w", stdout);
    printf("count %llu\n", sum);
    fclose(stdout);
    #endif
    //printf("count %llu\n", sum);
    //tmpTime.print("Counting time cost");
    //之后需要加上cudaFree

    // 尝试释放一些内存

    //gpuErrchk(cudaFree(dev_schedule->adj_mat));
    gpuErrchk(cudaFree(dev_schedule->father_prefix_id));
    gpuErrchk(cudaFree(dev_schedule->last));
    gpuErrchk(cudaFree(dev_schedule->next));
    gpuErrchk(cudaFree(dev_schedule->loop_set_prefix_id));
    gpuErrchk(cudaFree(dev_schedule->prefix_target));
    // gpuErrchk(cudaFree(dev_schedule->restrict_last));
    // gpuErrchk(cudaFree(dev_schedule->restrict_next));
    // gpuErrchk(cudaFree(dev_schedule->restrict_index));

    // gpuErrchk(cudaFree(dev_schedule->in_exclusion_optimize_vertex_id));
    // gpuErrchk(cudaFree(dev_schedule->in_exclusion_optimize_coef));
    // gpuErrchk(cudaFree(dev_schedule->in_exclusion_optimize_flag));

    gpuErrchk(cudaFree(dev_schedule));

    //delete[] edge_from;
    // delete[] in_exclusion_optimize_vertex_id;
    // delete[] in_exclusion_optimize_coef;
    // delete[] in_exclusion_optimize_flag;
    //delete[] only_need_size;
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

    gpuErrchk( cudaMalloc((void**)&dev_edge, size_edge));
    //gpuErrchk( cudaMalloc((void**)&dev_edge_from, size_edge));
    gpuErrchk( cudaMalloc((void**)&dev_labeled_vertex, size_labeled_vertex));
    gpuErrchk( cudaMalloc((void**)&dev_v_label, size_v_label));
    gpuErrchk( cudaMalloc((void**)&dev_tmp, size_tmp));
    gpuErrchk( cudaMalloc((void**)&dev_pattern_is_frequent_index, size_pattern_is_frequent_index));
    gpuErrchk( cudaMalloc((void**)&dev_is_frequent, size_is_frequent));
    gpuErrchk( cudaMalloc((void**)&dev_all_p_label, size_all_p_label));
    gpuErrchk( cudaMalloc((void**)&dev_label_start_idx, size_label_start_idx));

    gpuErrchk( cudaMemcpy(dev_edge, g->edge, size_edge, cudaMemcpyHostToDevice));
    //gpuErrchk( cudaMemcpy(dev_edge_from, edge_from, size_edge, cudaMemcpyHostToDevice));
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
        int threshold = 1;
        if(job_num > threshold) {
            fsm_cnt += pattern_matching_init(g, schedules[i], automorphisms, pattern_is_frequent_index[i], dev_is_frequent, dev_edge, dev_labeled_vertex, dev_v_label, dev_tmp, max_edge, job_num, dev_all_p_label, dev_fsm_set, dev_label_start_idx, min_support);
        }
        else {
            assert(false);
            fsm_cnt += g->fsm_vertex(0, schedules[i], all_p_label, automorphisms, is_frequent, pattern_is_frequent_index[i], max_edge, min_support, 16);
        }
        printf("temp fsm_cnt: %lld\n", fsm_cnt);
        mapping_start_idx_pos += schedules[i].get_size();
        if (get_pattern_edge_num(patterns[i]) != max_edge) //为了使得边数小于max_edge的pattern不被统计。正确性依赖于pattern按照边数排序
            fsm_cnt = 0;
        assert(pattern_is_frequent_index[i] % 32 == 0);
        assert(pattern_is_frequent_index[i + 1] % 32 == 0);
        int is_frequent_index = pattern_is_frequent_index[i] / 32;
        size_t is_frequent_size = (pattern_is_frequent_index[i + 1] - pattern_is_frequent_index[i]) / 32 * sizeof(uint32_t);
        gpuErrchk( cudaMemcpy(&is_frequent[is_frequent_index], &dev_is_frequent[is_frequent_index], is_frequent_size, cudaMemcpyDeviceToHost));
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
    // cudaSetDevice(2);

    LabeledGraph *g;
    DataLoader D;

    const std::string type = argv[1];
    const std::string path = argv[2];
    const int max_edge = atoi(argv[3]);
    const int min_support = atoi(argv[4]);

    DataType my_type;
    
    GetDataType(my_type, type);

    if(my_type == DataType::Invalid) {
        printf("Dataset not found!\n");
        return 0;
    }

    g = new LabeledGraph();
    assert(D.load_labeled_data(g,my_type,path.c_str())==true);
    fsm_init(g, max_edge, min_support);

    return 0;
}
