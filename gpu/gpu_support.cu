#include <graph.h>
#include <dataloader.h>
#include <vertex_set.h>
#include <common.h>
#include <schedule.h>
#include <motif_generator.h>

#include <cassert>
#include <cstring>
#include <cstdint>
#include <string>
#include <algorithm>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <sys/time.h>
#include <chrono>

#define likely(x)   __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

constexpr int THREADS_PER_BLOCK = 256;
constexpr int THREADS_PER_WARP = 32;
constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / THREADS_PER_WARP;

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

// struct GPUGroupDim2 {
//     int* data;
//     int size;
// };

// struct GPUGroupDim1 {
//     GPUGroupDim2* data;
//     int size;
// };

// struct GPUGroupDim0 {
//     GPUGroupDim1* data;
//     int size;
// };

class GPUSchedule {
public:
    inline __device__ int get_total_prefix_num() const { return total_prefix_num;}
    // inline __device__ int get_basic_prefix_num() const { return basic_prefix_num;}
    inline __device__ int get_father_prefix_id(int prefix_id) const { return father_prefix_id[prefix_id];}
    inline __device__ int get_loop_set_prefix_id(int loop) const { return loop_set_prefix_id[loop];}
    inline __device__ int get_size() const { return size;}
    inline __device__ int get_last(int i) const { return last[i];}
    inline __device__ int get_next(int i) const { return next[i];}
    // inline __device__ int get_break_size(int i) const { return break_size[i];}
    inline __device__ int get_in_exclusion_optimize_num() const { return in_exclusion_optimize_num;}
    // inline __device__ int get_total_restrict_num() const { return total_restrict_num;}
    inline __device__ int get_restrict_last(int i) const { return restrict_last[i];}
    inline __device__ int get_restrict_next(int i) const { return restrict_next[i];}
    inline __device__ int get_restrict_index(int i) const { return restrict_index[i];}

    // int* adj_mat;
    int* father_prefix_id;
    int* last;
    int* next;
    // int* break_size;
    int* loop_set_prefix_id;
    int* restrict_last;
    int* restrict_next;
    int* restrict_index;
    //bool* only_need_size;
    //int* in_exclusion_optimize_val;
    //GPUGroupDim0 in_exclusion_optimize_group;
    //int in_exclusion_optimize_val_size;
    int size;
    int total_prefix_num;
    //int basic_prefix_num;
    //int total_restrict_num;
    int in_exclusion_optimize_num;
    //int k_val;

    // int in_exclusion_optimize_vertex_id_size;
    // int* in_exclusion_optimize_vertex_id;
    // bool* in_exclusion_optimize_vertex_flag;
    // int* in_exclusion_optimize_vertex_coef;
    
    // int in_exclusion_optimize_array_size;
    // int* in_exclusion_optimize_coef;
    // bool* in_exclusion_optimize_flag;
    // int* in_exclusion_optimize_ans_pos;

    // uint32_t ans_array_offset;
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
    __device__ uint32_t operator[](int i) const { return data[i]; }
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

struct BitVector {
    __device__ void insert(uint32_t id) { //目前每个warp一个bitvector，且只有一个线程会操作，所以不加锁
        uint32_t i = id >> 5;
        uint32_t v = 1 << (id & 31);
        auto b = data[i];
        if ((b & v) == 0) {
            ++non_zero_cnt;
            data[i] = b | v;
        }
    }

    __device__ void insert_safe(uint32_t id) {
        uint32_t i = id >> 5;
        uint32_t v = 1 << (id & 31);
        auto b = atomicOr(&data[i], v);
        if ((b & v) == 0)
            atomicAdd(&non_zero_cnt, 1);
    }

    unsigned int non_zero_cnt;
    uint32_t* data;
};

__device__ unsigned int dev_cur_edge = 0; //用int表示边之后在大图上一定会出问题！

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
        
        if (found) {
            uint32_t offset = out_offset[lid] - 1;
            out[out_size + offset] = u;
        }

        if (lid == 0)
            out_size += out_offset[THREADS_PER_WARP - 1];
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

/**
 * @brief calculate | set0 - set1 |
 * @note set0 should be an ordered set, while set1 can be unordered
 * @todo rename 'subtraction' => 'difference'
 */
__device__ int unordered_subtraction_size(const GPUVertexSet& set0, const GPUVertexSet& set1, int size_after_restrict = -1)
{
    __shared__ int block_ret[WARPS_PER_BLOCK];

    int size0 = set0.get_size();
    int size1 = set1.get_size();
    if (size_after_restrict != -1)
        size0 = size_after_restrict;

    int wid = threadIdx.x / THREADS_PER_WARP;
    int lid = threadIdx.x % THREADS_PER_WARP;
    int &ret = block_ret[wid];
    if (lid == 0)
        ret = size0;
    __threadfence_block();

    int done1 = 0;
    while (done1 < size1)
    {
        if (lid + done1 < size1)
        {
            int l = 0, r = size0 - 1;
            uint32_t val = set1.get_data(lid + done1);
            //考虑之后换一下二分查找的写法，比如改为l < r，然后把mid的判断从循环里去掉，放到循环外(即最后l==r的时候)
            while (l <= r)
            {
                int mid = (l + r) >> 1;
                if (unlikely(set0.get_data(mid) == val))
                {
                    atomicSub(&ret, 1);
                    break;
                }
                if (set0.get_data(mid) < val)
                    l = mid + 1;
                else
                    r = mid - 1;
            }
            //binary search
        }
        done1 += THREADS_PER_WARP;
    }

    __threadfence_block();
    return ret;
}

/**
 * @brief 递归版本的pattern matching主函数。
 * @note 调用处初始深度为2（已经匹配了一条边对应的两个点）
 */
 /*
__device__ void GPU_pattern_matching_aggressive_func(const GPUSchedule* schedule, GPUVertexSet* vertex_set, GPUVertexSet& subtraction_set,
    GPUVertexSet& tmp_set, unsigned long long& local_ans, int depth, uint32_t *edge, uint32_t *vertex)
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
                    int tmp = unordered_subtraction_size(vertex_set[id], subtraction_set);
                    val = val * tmp;
                }
                else {
                    int id = loop_set_prefix_ids[cur_graph.data[cur_graph_rank].data[0]];
                    tmp_set.copy_from(vertex_set[id]);

                    for(int i = 1; i < cur_graph.data[cur_graph_rank].size; ++i) {
                        int id = loop_set_prefix_ids[cur_graph.data[cur_graph_rank].data[i]];
                        tmp_set.intersection_with(vertex_set[id]);
                    }
                    
                    int tmp = unordered_subtraction_size(tmp_set, subtraction_set);
                    val = val * tmp;
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
*/

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

constexpr int MAX_DEPTH = 8; // 非递归pattern matching支持的最大深度

template <int depth>
__device__ void GPU_pattern_matching_bruteforce(const GPUSchedule* schedule, GPUVertexSet* vertex_set, GPUVertexSet& subtraction_set,
    BitVector* fsm_sets, uint32_t *edge, uint32_t *vertex)
{
    int n = schedule->get_size();
    if (depth == n) {
        int wid = threadIdx.x % THREADS_PER_WARP;
        if (wid < n) // assume n <= 32
            fsm_sets[wid].insert_safe(subtraction_set[wid]);
        return;
    }

    int m = schedule->get_in_exclusion_optimize_num();
    if (depth == n - m) {
        int loop_set_prefix_ids[4]; // bad hard-coded value. (VLA is not supported on cuda)
        for (int i = 0; i < m; ++i)
            loop_set_prefix_ids[i] = schedule->get_loop_set_prefix_id(depth + i);
        
        bool opt_ok = true; // ok when all(set.size >= m for set in remaining_sets)
        for (int i = 0; i < m; ++i) {
            int set_size = unordered_subtraction_size(vertex_set[loop_set_prefix_ids[i]], subtraction_set);
            if (set_size < m) {
                opt_ok = false;
                break;
            }
        }
        if (opt_ok) {
            int wid = threadIdx.x % THREADS_PER_WARP;
            for (int i = 0; i < m; ++i) {
                auto &vertexes = vertex_set[loop_set_prefix_ids[i]];
                int sz = vertexes.get_size();
                for (int b = 0; b < sz; b += THREADS_PER_WARP) {
                    int j = b + wid;
                    if (j < sz && !subtraction_set.has_data(vertexes[j])) {
                        fsm_sets[depth + i].insert_safe(vertexes[j]);
                    }
                }
            }
            if (wid < depth) // assume n <= 32
                fsm_sets[wid].insert_safe(subtraction_set[wid]);
            return;
        }
    }

    int loop_set_prefix_id = schedule->get_loop_set_prefix_id(depth);
    int loop_size = vertex_set[loop_set_prefix_id].get_size();
    if (loop_size <= 0) //这个判断可能可以删了
        return;

    uint32_t* loop_data_ptr = vertex_set[loop_set_prefix_id].get_data_ptr();
    uint32_t min_vertex = 0xffffffff;
    for (int i = schedule->get_restrict_last(depth); i != -1; i = schedule->get_restrict_next(i))
        if (min_vertex > subtraction_set.get_data(schedule->get_restrict_index(i)))
            min_vertex = subtraction_set.get_data(schedule->get_restrict_index(i));

    for (int i = 0; i < loop_size; ++i) {
        uint32_t v = loop_data_ptr[i];
        if (min_vertex <= v)
            break;
        if (subtraction_set.has_data(v))
            continue;
        unsigned int l, r;
        get_edge_index(v, l, r);
        bool is_zero = false;
        for (int prefix_id = schedule->get_last(depth); prefix_id != -1; prefix_id = schedule->get_next(prefix_id)) {
            vertex_set[prefix_id].build_vertex_set(schedule, vertex_set, &edge[l], r - l, prefix_id);
            if (vertex_set[prefix_id].get_size() == 0) {
                is_zero = true;
                break;
            }
        }
        if (is_zero)
            continue;
        if (depth + 1 < MAX_DEPTH) {
            if (threadIdx.x % THREADS_PER_WARP == 0)
                subtraction_set.push_back(v);
            __threadfence_block();
        }
        GPU_pattern_matching_bruteforce<depth + 1>(schedule, vertex_set, subtraction_set, fsm_sets, edge, vertex);
        if (depth + 1 < MAX_DEPTH) {
            if (threadIdx.x % THREADS_PER_WARP == 0)
                subtraction_set.pop_back();
            __threadfence_block();
        }
    }
}

template <>
__device__ void GPU_pattern_matching_bruteforce<MAX_DEPTH>(const GPUSchedule* schedule, GPUVertexSet* vertex_set, GPUVertexSet& subtraction_set,
    BitVector* fsm_sets, uint32_t *edge, uint32_t *vertex)
{
    int n = schedule->get_size();
    if (MAX_DEPTH == n) {
        int wid = threadIdx.x % THREADS_PER_WARP;
        if (wid < n)
            fsm_sets[wid].insert_safe(subtraction_set[wid]);
    }
}

/**
 * @note `buffer_size`实际上是每个节点的最大邻居数量，而非所用空间大小
 */
__global__ void gpu_pattern_matching(uint32_t edge_num, uint32_t buffer_size, uint32_t *edge_from, uint32_t *edge, uint32_t *vertex,
    uint32_t *tmp, BitVector* fsm_sets, const GPUSchedule* schedule) {
    __shared__ unsigned int block_edge_idx[WARPS_PER_BLOCK]; //用int表示边之后在大图上一定会出问题！
    //之后考虑把tmp buffer都放到shared里来（如果放得下）
    extern __shared__ GPUVertexSet block_vertex_set[];
    
    int num_prefixes = schedule->get_total_prefix_num();
    int num_vertex_sets_per_warp = num_prefixes + 1;

    int wid = threadIdx.x / THREADS_PER_WARP; // warp id within the block
    int lid = threadIdx.x % THREADS_PER_WARP; // lane id
    int global_wid = blockIdx.x * WARPS_PER_BLOCK + wid; // global warp id
    unsigned int &edge_idx = block_edge_idx[wid];
    GPUVertexSet *vertex_set = block_vertex_set + wid * num_vertex_sets_per_warp;

    auto block_fsm_sets = &fsm_sets[blockIdx.x * schedule->get_size()];

    if (lid == 0) {
        edge_idx = 0;
        uint32_t offset = buffer_size * global_wid * num_vertex_sets_per_warp;
        for (int i = 0; i < num_vertex_sets_per_warp; ++i)
        {
            vertex_set[i].set_data_ptr(tmp + offset); // 注意这是个指针+整数运算，自带*4
            offset += buffer_size;
        }
    }
    GPUVertexSet& subtraction_set = vertex_set[num_prefixes];

    __threadfence_block(); //之后考虑把所有的syncthreads都改成syncwarp


    uint32_t v0, v1;
    uint32_t l, r;

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
        if (i >= edge_num) break;
       
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
        
        GPU_pattern_matching_bruteforce<2>(schedule, vertex_set, subtraction_set, block_fsm_sets, edge, vertex);
    }
}

__global__ void merge_bitmaps(uint32_t block_bitmaps[], uint32_t final_bitmap[], int schedule_size, int nr_bitmap_elements)
{
    int bid = blockIdx.x;
    int bitmap_group_elem_count = nr_bitmap_elements * schedule_size;
    auto bitmap_base = block_bitmaps + bid * bitmap_group_elem_count;
    for (int b = 0; b < bitmap_group_elem_count; b += THREADS_PER_BLOCK) {
        int k = b + threadIdx.x;
        if (k < bitmap_group_elem_count) {
            auto v = bitmap_base[k];
            atomicOr(&final_bitmap[k], v);
        }
    }
}

__global__ void get_bitmap_sizes(BitVector controls[], uint32_t final_bitmap[], int schedule_size, int nr_bitmap_elements)
{
    size_t single_bitmap_bits = 1ul * nr_bitmap_elements * sizeof(uint32_t) * 8;
    size_t total_bits = single_bitmap_bits * schedule_size;
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < total_bits) {
        int i = id / single_bitmap_bits;
        auto v = final_bitmap[id / 32];
        if (v & (1 << (id % 32)))
            atomicAdd(&(controls[i].non_zero_cnt), 1);
    }
}

void pattern_matching_init(Graph *g, const Schedule& schedule) {
    int num_blocks = 1024;
    int num_total_warps = num_blocks * WARPS_PER_BLOCK;

    int schedule_size = schedule.get_size();
    size_t size_edge = g->e_cnt * sizeof(uint32_t);
    size_t size_vertex = (g->v_cnt + 1) * sizeof(uint32_t);
    size_t size_tmp = VertexSet::max_intersection_size * sizeof(uint32_t) * num_total_warps * (schedule.get_total_prefix_num() + 1); //prefix + subtraction
    
    int nr_bitmap_elements = (g->v_cnt + 31) / 32;
    size_t single_bitmap_size = nr_bitmap_elements * sizeof(uint32_t);
    size_t size_all_bitmap = schedule_size * num_blocks * single_bitmap_size; // every block has a n bitmaps
    size_t size_final_bitmap = schedule_size * single_bitmap_size;

    uint32_t *edge_from = new uint32_t[g->e_cnt];
    for (uint32_t i = 0; i < g->v_cnt; ++i)
        for (uint32_t j = g->vertex[i]; j < g->vertex[i+1]; ++j)
            edge_from[j] = i;

    uint32_t *dev_edge;
    uint32_t *dev_edge_from;
    uint32_t *dev_vertex;
    uint32_t *dev_tmp;
    uint32_t *dev_bitmap, *dev_final_bitmap;
    BitVector *fsm_sets, *final_fsm_sets;

    gpuErrchk( cudaMalloc(&dev_bitmap, size_all_bitmap));
    gpuErrchk( cudaMemset(dev_bitmap, 0, size_all_bitmap));
    gpuErrchk( cudaMalloc(&dev_final_bitmap, size_final_bitmap));
    gpuErrchk( cudaMemset(dev_final_bitmap, 0, size_final_bitmap));

    gpuErrchk( cudaMallocManaged(&fsm_sets, sizeof(BitVector) * schedule_size * num_blocks));
    gpuErrchk( cudaMallocManaged(&final_fsm_sets, sizeof(BitVector) * schedule_size));

    for (int i = 0; i < schedule_size; ++i) {
        final_fsm_sets[i].non_zero_cnt = 0;
        final_fsm_sets[i].data = dev_final_bitmap + i * nr_bitmap_elements;
    }

    for (int i = 0; i < num_blocks; ++i) {
        for (int j = 0; j < schedule_size; ++j) {
            int k = i * schedule_size + j;
            fsm_sets[k].non_zero_cnt = 0;
            fsm_sets[k].data = dev_bitmap + k * nr_bitmap_elements;
        }
    }

    gpuErrchk( cudaMalloc((void**)&dev_edge, size_edge));
    gpuErrchk( cudaMalloc((void**)&dev_edge_from, size_edge));
    gpuErrchk( cudaMalloc((void**)&dev_vertex, size_vertex));
    gpuErrchk( cudaMalloc((void**)&dev_tmp, size_tmp));

    gpuErrchk( cudaMemcpy(dev_edge, g->edge, size_edge, cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy(dev_edge_from, edge_from, size_edge, cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy(dev_vertex, g->vertex, size_vertex, cudaMemcpyHostToDevice));

    unsigned int cur_edge = 0;
    gpuErrchk( cudaMemcpyToSymbol(dev_cur_edge, &cur_edge, sizeof(cur_edge)));

    //memcpy schedule
    GPUSchedule* dev_schedule;
    gpuErrchk( cudaMallocManaged((void**)&dev_schedule, sizeof(GPUSchedule)));
    int max_prefix_num = schedule_size * (schedule_size - 1) / 2;

    gpuErrchk( cudaMallocManaged((void**)&dev_schedule->father_prefix_id, sizeof(int) * max_prefix_num));
    gpuErrchk( cudaMemcpy(dev_schedule->father_prefix_id, schedule.get_father_prefix_id_ptr(), sizeof(int) * max_prefix_num, cudaMemcpyHostToDevice));

    gpuErrchk( cudaMallocManaged((void**)&dev_schedule->last, sizeof(int) * schedule_size));
    gpuErrchk( cudaMemcpy(dev_schedule->last, schedule.get_last_ptr(), sizeof(int) * schedule_size, cudaMemcpyHostToDevice));

    gpuErrchk( cudaMallocManaged((void**)&dev_schedule->next, sizeof(int) * max_prefix_num));
    gpuErrchk( cudaMemcpy(dev_schedule->next, schedule.get_next_ptr(), sizeof(int) * max_prefix_num, cudaMemcpyHostToDevice));
    
    
    gpuErrchk( cudaMallocManaged((void**)&dev_schedule->loop_set_prefix_id, sizeof(int) * schedule_size));
    gpuErrchk( cudaMemcpy(dev_schedule->loop_set_prefix_id, schedule.get_loop_set_prefix_id_ptr(), sizeof(int) * schedule_size, cudaMemcpyHostToDevice));

    gpuErrchk( cudaMallocManaged((void**)&dev_schedule->restrict_last, sizeof(int) * schedule_size));
    gpuErrchk( cudaMemcpy(dev_schedule->restrict_last, schedule.get_restrict_last_ptr(), sizeof(int) * schedule_size, cudaMemcpyHostToDevice));
    
    gpuErrchk( cudaMallocManaged((void**)&dev_schedule->restrict_next, sizeof(int) * max_prefix_num));
    gpuErrchk( cudaMemcpy(dev_schedule->restrict_next, schedule.get_restrict_next_ptr(), sizeof(int) * max_prefix_num, cudaMemcpyHostToDevice));
    
    gpuErrchk( cudaMallocManaged((void**)&dev_schedule->restrict_index, sizeof(int) * max_prefix_num));
    gpuErrchk( cudaMemcpy(dev_schedule->restrict_index, schedule.get_restrict_index_ptr(), sizeof(int) * max_prefix_num, cudaMemcpyHostToDevice));

    dev_schedule->size = schedule_size;
    dev_schedule->total_prefix_num = schedule.get_total_prefix_num();
    dev_schedule->in_exclusion_optimize_num = schedule.get_in_exclusion_optimize_num();

    printf("schedule.prefix_num: %d\n", schedule.get_total_prefix_num());
    printf("shared memory for vertex set per block: %ld bytes\n", 
        (schedule.get_total_prefix_num() + 1) * WARPS_PER_BLOCK * sizeof(GPUVertexSet));

    uint32_t buffer_size = VertexSet::max_intersection_size;
    uint32_t block_shmem_size = (schedule.get_total_prefix_num() + 1) * WARPS_PER_BLOCK * sizeof(GPUVertexSet);
    // 注意：此处没有错误，buffer_size代指每个顶点集所需的int数目，无需再乘sizeof(uint32_t)，但是否考虑对齐？
    //因为目前用了managed开内存，所以第一次运行kernel会有一定额外开销，考虑运行两次，第一次作为warmup
    
    int max_active_blocks_per_sm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks_per_sm, gpu_pattern_matching, THREADS_PER_BLOCK, block_shmem_size);
    printf("max number of active warps per SM: %d\n", max_active_blocks_per_sm * WARPS_PER_BLOCK);
    
    using namespace std::chrono;
    auto t1 = system_clock::now();
    fflush(stdout);

    gpu_pattern_matching<<<num_blocks, THREADS_PER_BLOCK, block_shmem_size>>>
        (g->e_cnt, buffer_size, dev_edge_from, dev_edge, dev_vertex, dev_tmp, fsm_sets, dev_schedule);
    
    merge_bitmaps<<<num_blocks, THREADS_PER_BLOCK>>>(dev_bitmap, dev_final_bitmap, schedule_size, nr_bitmap_elements);

    size_t total_bits = 1ul * schedule_size * nr_bitmap_elements * sizeof(uint32_t) * 8; 
    get_bitmap_sizes<<<(total_bits + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>
        (final_fsm_sets, dev_final_bitmap, schedule_size, nr_bitmap_elements);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    int support = g->v_cnt;
    for (int i = 0; i < schedule_size; ++i) {
        int set_size = final_fsm_sets[i].non_zero_cnt;
        printf("pattern vertex %d: set size = %d\n", i, set_size);
        if (support > set_size)
            support = set_size;
    }

    auto t2 = system_clock::now();
    auto elapsed = duration_cast<microseconds>(t2 - t1).count() * 1e-6;
    printf("support = %d (%g seconds)\n", support, elapsed);

    // 尝试释放一些内存
    gpuErrchk(cudaFree(dev_edge));
    gpuErrchk(cudaFree(dev_edge_from));
    gpuErrchk(cudaFree(dev_vertex));
    gpuErrchk(cudaFree(dev_tmp));
    gpuErrchk(cudaFree(dev_bitmap));
    gpuErrchk(cudaFree(dev_final_bitmap))
    gpuErrchk(cudaFree(fsm_sets));
    gpuErrchk(cudaFree(final_fsm_sets));

    gpuErrchk(cudaFree(dev_schedule->father_prefix_id));
    gpuErrchk(cudaFree(dev_schedule->last));
    gpuErrchk(cudaFree(dev_schedule->next));
    gpuErrchk(cudaFree(dev_schedule->loop_set_prefix_id));
    gpuErrchk(cudaFree(dev_schedule->restrict_last));
    gpuErrchk(cudaFree(dev_schedule->restrict_next));
    gpuErrchk(cudaFree(dev_schedule->restrict_index));

    gpuErrchk(cudaFree(dev_schedule));

    delete[] edge_from;
}

int main(int argc,char *argv[]) {
    Graph *g;
    DataLoader D;

    using std::chrono::system_clock;
    auto t1 = system_clock::now();

    bool ok = D.fast_load(g, argv[1]);
    if (!ok) {
        printf("data load failure :-(\n");
        return 0;
    }

    auto t2 = system_clock::now();
    auto load_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    printf("Load data success! time: %g seconds\n", load_time.count() / 1.0e6);
    fflush(stdout);

    // const char *pattern_str = "0111010011100011100001100"; // 5 house p1
    //const char *pattern_str = "011011101110110101011000110000101000"; // 6 p2
    // const char *pattern_str = "0111111101111111011101110100111100011100001100000"; // 7 p5
    // const char *pattern_str = "0111111101111111011001110100111100011000001100000"; // 7 p6

    int pattern_size = atoi(argv[2]);
    const char *adj_mat = argv[3];

    Pattern p(pattern_size, adj_mat);
    bool is_pattern_valid;
    Schedule schedule(p, is_pattern_valid, 1, 1, true, g->v_cnt, g->e_cnt, g->tri_cnt);
    if (!is_pattern_valid) {
        printf("pattern is invalid!\n");
        return 0;
    }

    pattern_matching_init(g, schedule);

    return 0;
}
