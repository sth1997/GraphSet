/**
 * 这个版本里面没有细粒度计时。有计时的在gpu_graph_with_timer.cu里面。
 * 而且计时的方式与zms版本略有区别。
 */
#include <graph.h>
#include <dataloader.h>
#include <vertex_set.h>
#include <common.h>

#include <cassert>
#include <cstdint>
#include <string>
#include <algorithm>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <sys/time.h>
#include <chrono>
//#define DO_INTERSECTION_128

constexpr int THREADS_PER_BLOCK = 64;
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
static __device__ uint32_t do_intersection(uint32_t*, const uint32_t*, const uint32_t*, uint32_t, uint32_t);

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
constexpr int MAX_INTERSECTION_CONCURRENCY = 128; // 一个warp一次做128个a数组中的元素（4*32），减少求前缀和的时间
//目前性能较差（个人认为是shared memory读写次数较多），暂时别用这个函数，之后优化
__device__ uint32_t do_intersection_more_concurrency(uint32_t* out, const uint32_t* a, const uint32_t* b, uint32_t na, uint32_t nb)
{
    //__shared__ uint32_t block_out_offset[MAX_INTERSECTION_CONCURRENCY * WARPS_PER_BLOCK];
    __shared__ uint32_t block_tmp_prefix_sum[THREADS_PER_WARP * WARPS_PER_BLOCK];
    __shared__ uint32_t block_out_size[WARPS_PER_BLOCK];
    __shared__ uint32_t out_offset_buf_block[(MAX_INTERSECTION_CONCURRENCY >> 5) * WARPS_PER_BLOCK]; //32>>5=1, 64>>5=2, 128>>5=4
    __shared__ uint32_t u_block[(MAX_INTERSECTION_CONCURRENCY >> 5) * WARPS_PER_BLOCK];
    __shared__ bool found_block[(MAX_INTERSECTION_CONCURRENCY >> 5) * WARPS_PER_BLOCK];

    int wid = threadIdx.x / THREADS_PER_WARP; // warp id
    int lid = threadIdx.x % THREADS_PER_WARP; // lane id
    //uint32_t *out_offset = block_out_offset + wid * THREADS_PER_WARP;
    //uint32_t *out_offset = block_out_offset + wid * MAX_INTERSECTION_CONCURRENCY;
    uint32_t* tmp_prefix_sum = block_tmp_prefix_sum + wid * THREADS_PER_WARP;
    uint32_t &out_size = block_out_size[wid];
    uint32_t* out_offset_buf = out_offset_buf_block + (MAX_INTERSECTION_CONCURRENCY >> 5) * wid; //32>>5=1, 64>>5=2, 128>>5=4
    uint32_t* u = u_block + (MAX_INTERSECTION_CONCURRENCY >> 5) * wid;
    bool* found = found_block + (MAX_INTERSECTION_CONCURRENCY >> 5) * wid;

    if (lid == 0)
        out_size = 0;
    int concurrency = THREADS_PER_WARP;
    int shift = 0;
    if (na > THREADS_PER_WARP)
    {
        if (na <= 64)
        {
            concurrency = 64;
            shift = 1;
        }
        else
        {
            concurrency = MAX_INTERSECTION_CONCURRENCY;
            shift = 2;
        }
    }

    uint32_t v, num_done = 0;
    
    while (num_done < na) {
        int start_offset = (lid << shift);
        //for (int start_offset = lid; start_offset < concurrency; start_offset += THREADS_PER_WARP)
        /*for (int i = 0; i < (1 << shift); ++i)
            if (start_offset + i + num_done < na)
                u[i] = a[start_offset + i + num_done]; // u: an element in set a*/
        for (int i = 0; i < (1 << shift); ++i)
        {
            found[i] = 0;
            if (start_offset + i + num_done < na) {
                u[i] = a[start_offset + i + num_done]; // u: an element in set a
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
                    if (b[mid] < u[i]) {
                        l = mid + 1;
                    } else if (b[mid] > u[i]) {
                        r = mid - 1;
                    } else {
                        found[i] = 1;
                        break;
                    }
                }
            }
            out_offset_buf[i] = found[i];
        }
        uint32_t tmp_sum = 0;
        //for (int i = (lid << shift); i < ((lid + 1) << shift); ++i)
        //for (int i = 0; i < (1 << shift); ++i)
        //    out_offset_buf[i] = out_offset[start_offset + i];
        for (int i = 0; i < (1 << shift); ++i)
            tmp_sum += out_offset_buf[i];
        tmp_prefix_sum[lid] = tmp_sum;
        __threadfence_block();
        for (int s = 1; s < THREADS_PER_WARP; s *= 2) {
            uint32_t v = lid >= s ? tmp_prefix_sum[lid - s] : 0; //TODO: 考虑之后直接用寄存器shuffle而不是shared memory
            __threadfence_block();
            tmp_prefix_sum[lid] += v;
            __threadfence_block();
        }
        if (lid == 0)
            tmp_sum = 0;
        else
            tmp_sum = tmp_prefix_sum[lid - 1];
        /*for (int i = (lid << shift); i < ((lid + 1) << shift); ++i)
        {
            tmp_sum += out_offset[i];//TODO: 这里有比较严重的bank conflict，考虑之后能不能用位运算加速？
            out_offset[i] = tmp_sum;
        }*/
        for (int i = 0; i < (1 << shift); ++i) // 这个可以和下一个循环合并，只不过可能会有数据依赖而stall，之后考虑一下
        {
            tmp_sum += out_offset_buf[i];
            out_offset_buf[i] = tmp_sum;
        }

        /*bool found;
        if (lid == 0)
            found = out_offset[0];
        else
            found = out_offset[lid] > out_offset[lid - 1];
        if (found)
            out[out_size + out_offset[lid] - 1] = a[num_done + lid];
        for (int start_offset = lid + THREADS_PER_WARP; start_offset < concurrency; start_offset += THREADS_PER_WARP) //如果之后固定128，把这里展开会性能更好
        {
            if (out_offset[start_offset] > out_offset[start_offset - 1]) {
                uint32_t offset = out_offset[start_offset] - 1;
                out[out_size + offset] = a[num_done + start_offset];
            }
        }

        if (lid == 0)
            //out_size += tmp_prefix_sum[THREADS_PER_WARP - 1];
            out_size += out_offset[THREADS_PER_WARP - 1];*/
        for (int i = 0; i < (1 << shift); ++i)
        {
            if (found[i])
            {
                uint32_t offset = out_offset_buf[i] - 1;
                out[out_size + offset] = u[i];
            }
        }
        if (lid == 31)
            out_size += out_offset_buf[(1 << shift) - 1];
        num_done += concurrency;
    }

    __threadfence_block();
    return out_size;
}

#ifdef DO_INTERSECTION_128
__device__ uint32_t do_intersection_128(uint32_t* out, const uint32_t* a, const uint32_t* b, uint32_t na, uint32_t nb)
{
    //__shared__ uint32_t block_out_offset[MAX_INTERSECTION_CONCURRENCY * WARPS_PER_BLOCK];
    __shared__ uint32_t block_tmp_prefix_sum[THREADS_PER_WARP * WARPS_PER_BLOCK];
    __shared__ uint32_t block_out_size[WARPS_PER_BLOCK];
    uint32_t out_offset0, out_offset1, out_offset2, out_offset3;
    uint32_t u0, u1, u2, u3;
    bool found0, found1, found2, found3;

    int wid = threadIdx.x / THREADS_PER_WARP; // warp id
    int lid = threadIdx.x % THREADS_PER_WARP; // lane id
    uint32_t* tmp_prefix_sum = block_tmp_prefix_sum + wid * THREADS_PER_WARP;
    uint32_t &out_size = block_out_size[wid];

    if (lid == 0)
        out_size = 0;
    //constexpr int concurrency = 128;
    //constexpr int shift = 2;

    uint32_t v, num_done = 0;
    int start_offset = (lid << 2);

    //while (num_done < na) {
        found0 = found1 = found2 = found3 = 0;
        if (start_offset < na) {
            u0 = a[start_offset]; // u: an element in set a
            int mid, l = 0, r = int(nb) - 1;
            while (l <= r) {
                mid = (l + r) >> 1;
                if (b[mid] < u0) {
                    l = mid + 1;
                } else if (b[mid] > u0) {
                    r = mid - 1;
                } else {
                    found0 = 1;
                    break;
                }
            }
        }
        out_offset0 = found0;
        if (start_offset + 1 < na) {
            u1 = a[start_offset + 1]; // u: an element in set a
            int mid, l = 0, r = int(nb) - 1;
            while (l <= r) {
                mid = (l + r) >> 1;
                if (b[mid] < u1) {
                    l = mid + 1;
                } else if (b[mid] > u1) {
                    r = mid - 1;
                } else {
                    found1 = 1;
                    break;
                }
            }
        }
        out_offset1 = found1;
        if (start_offset + 2 < na) {
            u2 = a[start_offset + 2]; // u: an element in set a
            int mid, l = 0, r = int(nb) - 1;
            while (l <= r) {
                mid = (l + r) >> 1;
                if (b[mid] < u2) {
                    l = mid + 1;
                } else if (b[mid] > u2) {
                    r = mid - 1;
                } else {
                    found2 = 1;
                    break;
                }
            }
        }
        out_offset2 = found2;
        if (start_offset + 3 < na) {
            u3 = a[start_offset + 3]; // u: an element in set a
            int mid, l = 0, r = int(nb) - 1;
            while (l <= r) {
                mid = (l + r) >> 1;
                if (b[mid] < u3) {
                    l = mid + 1;
                } else if (b[mid] > u3) {
                    r = mid - 1;
                } else {
                    found3 = 1;
                    break;
                }
            }
        }
        out_offset3 = found3;

        uint32_t tmp_sum = (out_offset0 + out_offset1) + (out_offset2 + out_offset3);
        tmp_prefix_sum[lid] = tmp_sum;
        __threadfence_block();
        for (int s = 1; s < THREADS_PER_WARP; s *= 2) {
            uint32_t v = lid >= s ? tmp_prefix_sum[lid - s] : 0; //TODO: 考虑之后直接用寄存器shuffle而不是shared memory
            __threadfence_block();
            tmp_prefix_sum[lid] += v;
            __threadfence_block();
        }
        if (lid == 0)
            tmp_sum = 0;
        else
            tmp_sum = tmp_prefix_sum[lid - 1];
        
        tmp_sum += out_offset0;//这几行可以直接用out_offset+=，不需要tmp_sum
        out_offset0 = tmp_sum;
        tmp_sum += out_offset1;
        out_offset1 = tmp_sum;
        tmp_sum += out_offset2;
        out_offset2 = tmp_sum;
        tmp_sum += out_offset3;
        out_offset3 = tmp_sum;

        if (found0)
            out[out_size - 1 + out_offset0] = u0;
        if (found1)
            out[out_size - 1 + out_offset1] = u1;
        if (found2)
            out[out_size - 1 + out_offset2] = u2;
        if (found3)
            out[out_size - 1 + out_offset3] = u3;

        if (lid == 31)
            out_size += out_offset3;
    /*    num_done += 128;
        start_offset += 128;
    }*/

    __threadfence_block();
    return out_size;
}
#endif

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

    uint32_t v, num_done = 0;
    #ifdef DO_INTERSECTION_128
    while (na - num_done >= 128)
    {
        uint32_t ret = do_intersection_128(out, a, b, na, nb);
        if (lid == 0)
            out_size += ret;
        num_done += 128;
    }
    #endif
    while (num_done < na) {
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
        num_done += THREADS_PER_WARP;
    }

    __threadfence_block();
    return out_size;
}

/**
 * search-based intersection
 * 
 * returns the size of the intersection set
 * 
 * @note：只返回交集大小，并不实际地求出交集。
            集合中的元素的最后一位为flag，flag为0表示元素合法（即每个元素是原始节点编号的2倍或者2倍+1）。
 * @todo：shared memory缓存优化
 */
__device__ uint32_t get_intersection_size_with_flag(const uint32_t* a, const uint32_t* b, uint32_t na, uint32_t nb)
{
    __shared__ uint32_t block_out_size[WARPS_PER_BLOCK];

    int wid = threadIdx.x / THREADS_PER_WARP; // warp id
    int lid = threadIdx.x % THREADS_PER_WARP; // lane id
    uint32_t &out_size = block_out_size[wid];

    if (lid == 0)
        out_size = 0;

    uint32_t num_done = 0;
    while (num_done < na) {
        uint32_t u = 0;
        if (num_done + lid < na) {
            u = a[num_done + lid];
            if ((u & 1) == 0) //只使用a中合法元素查找，这样求交结果也一定都是合法元素
            {
                int mid, l = 0, r = int(nb) - 1;
                while (l <= r) {
                    mid = (l + r) >> 1;
                    if (b[mid] < u) {
                        l = mid + 1;
                    } else if (b[mid] > u) {
                        r = mid - 1;
                    } else {
                        atomicAdd(&out_size, 1);
                        break;
                    }
                }
            }
        }
        __threadfence_block();
        num_done += THREADS_PER_WARP;
    }

    __threadfence_block();
    return out_size;
}

/**
 * search-based intersection
 * 
 * returns the size of the intersection set
 * 
 * @note：a和b并不是地位相等的。如果要进行in-place操作，请把输入放在a而不是b。
            集合中的元素的最后一位为flag，flag为0表示元素合法（即每个元素是原始节点编号的2倍或者2倍+1）。
 * @todo：shared memory缓存优化
 */
__device__ uint32_t do_intersection_with_flag(uint32_t* out, const uint32_t* a, const uint32_t* b, uint32_t na, uint32_t nb)
{
    __shared__ uint32_t block_out_offset[THREADS_PER_BLOCK];
    __shared__ uint32_t block_out_size[WARPS_PER_BLOCK];

    int wid = threadIdx.x / THREADS_PER_WARP; // warp id
    int lid = threadIdx.x % THREADS_PER_WARP; // lane id
    uint32_t *out_offset = block_out_offset + wid * THREADS_PER_WARP;
    uint32_t &out_size = block_out_size[wid];

    if (lid == 0)
        out_size = 0;

    uint32_t num_done = 0;
    while (num_done < na) {
        bool found = 0;
        uint32_t u = 0;
        if (num_done + lid < na) {
            u = a[num_done + lid];
            if ((u & 1) == 0) //只使用a中合法元素查找，这样求交结果也一定都是合法元素
            {
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
        }
        out_offset[lid] = found;
        __threadfence_block();

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
        num_done += THREADS_PER_WARP;
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
 * @brief calculate set0 = set0 - set1。返回集合大小。输入的set0元素必须是原节点编号*2（不允许有单数存在），set1为原节点编号。
 * @note set0 should be an ordered set, while set1 can be unordered。元素不被真正地删除，而是被+1（最低位为1，表示不合法）。需要注意，得到的set0的size不变，仍然包含被删除元素。
            注意区别size0和ret的区别。ret是真实差集的大小，而size0是占用存储空间。也就是说元素不被真正的删除，size0不会变。之后如果用set0继续求交集，需要用size0而不是ret当做元素个数。
 * @todo rename 'subtraction' => 'difference'
 */
__device__ int unordered_subtraction_with_flag(uint32_t* lbases, const uint32_t* rbases, int size0, int size1, int size_after_restrict = -1)
{
    if (size_after_restrict != -1)
        size0 = size_after_restrict;

    __shared__ uint32_t block_out_size[WARPS_PER_BLOCK];

    int wid = threadIdx.x / THREADS_PER_WARP;
    int lid = threadIdx.x % THREADS_PER_WARP;
    uint32_t &ret = block_out_size[wid];
    if (lid == 0)
        ret = size0;
    __threadfence_block();

    int done1 = lid;
    while (done1 < size1)
    {
        int l = 0, r = size0 - 1;
        uint32_t val = rbases[done1] << 1;
        //考虑之后换一下二分查找的写法，比如改为l < r，然后把mid的判断从循环里去掉，放到循环外(即最后l==r的时候)
        while (l <= r)
        {
            int mid = (l + r) >> 1;
            if (lbases[mid] < val)
                l = mid + 1;
            else if (lbases[mid] > val)
                r = mid - 1;
            else
            {
                ++lbases[mid];
                atomicSub(&ret, 1);
                break;
            }
        }
        done1 += THREADS_PER_WARP;
    }
    __threadfence_block();
    return ret;
}

//默认所有set1的都相同（所以rbases和size1不是数组）
constexpr int MAX_SUBTRACTION_CNT = 8; //同时做8个subtraction
__device__ int* unordered_subtraction_with_flag_parallel(uint32_t* lbases[], const uint32_t* rbases, int size0[], int size1, int subtraction_cnt)
{
    //if (size_after_restrict != -1)
    //    size0 = size_after_restrict;

    __shared__ int block_ret[WARPS_PER_BLOCK * MAX_SUBTRACTION_CNT];

    int wid = threadIdx.x / THREADS_PER_WARP;
    int lid = threadIdx.x % THREADS_PER_WARP;
    int total_size1 = size1 * subtraction_cnt;
    int *ret = block_ret + wid * MAX_SUBTRACTION_CNT;
    if (lid < subtraction_cnt)
        ret[lid] = size0[lid];
    __threadfence_block();

    int done1 = lid;
    while (done1 < total_size1)
    {
        int subtraction_index = done1 / size1;
        uint32_t* lbase = lbases[subtraction_index];
        int l = 0, r = size0[subtraction_index] - 1;
        uint32_t val = rbases[done1 % size1] << 1;
        //考虑之后换一下二分查找的写法，比如改为l < r，然后把mid的判断从循环里去掉，放到循环外(即最后l==r的时候)
        while (l <= r)
        {
            int mid = (l + r) >> 1;
            if (lbase[mid] < val)
                l = mid + 1;
            else if (lbase[mid] > val)
                r = mid - 1;
            else
            {
                ++lbase[mid];
                atomicSub(&ret[subtraction_index], 1);
                break;
            }
        }
        done1 += THREADS_PER_WARP;
    }
    __threadfence_block();
    return ret;
}

/**
 * @brief calculate | set0 - set1 |
 * @note set0 should be an ordered set, while set1 can be unordered
 * @todo rename 'subtraction' => 'difference'
 */
__device__ int unordered_subtraction_size(const uint32_t* lbases, const uint32_t* rbases, int size0, int size1, int size_after_restrict = -1)
{
    __shared__ int block_ret[WARPS_PER_BLOCK];

    if (size_after_restrict != -1)
        size0 = size_after_restrict;

    int wid = threadIdx.x / THREADS_PER_WARP;
    int lid = threadIdx.x % THREADS_PER_WARP;
    int &ret = block_ret[wid];
    if (lid == 0)
        ret = size0;

    //TODO: 把done1初始化为lid，之后就可以省掉lid + done1的运算。把这个优化用到intersection里不知道会不会有问题（比如只有一部分线程求前缀和可能有问题？）
    int done1 = 0;
    while (done1 < size1)
    {
        if (lid + done1 < size1)
        {
            int l = 0, r = size0 - 1;
            uint32_t val = rbases[lid + done1];
            //考虑之后换一下二分查找的写法，比如改为l < r，然后把mid的判断从循环里去掉，放到循环外(即最后l==r的时候)
            while (l <= r)
            {
                int mid = (l + r) >> 1;
                if (lbases[mid] == val)//TODO：命中的概率是很低的，所以放在最后一个else会不会性能更好？
                {
                    atomicSub(&ret, 1);
                    break;
                }
                if (lbases[mid] < val)
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

__device__ int* unordered_subtraction_parallel(uint32_t* lbases[], const uint32_t* rbases, int size0[], int size1, int subtraction_cnt)
{
    //if (size_after_restrict != -1)
    //    size0 = size_after_restrict;

    __shared__ int block_ret[WARPS_PER_BLOCK * MAX_SUBTRACTION_CNT];

    int wid = threadIdx.x / THREADS_PER_WARP;
    int lid = threadIdx.x % THREADS_PER_WARP;
    int total_size1 = size1 * subtraction_cnt;
    int *ret = block_ret + wid * MAX_SUBTRACTION_CNT;
    if (lid < subtraction_cnt)
        ret[lid] = size0[lid];
    __threadfence_block();

    int done1 = lid;
    while (done1 < total_size1)
    {
        int subtraction_index = done1 / size1;
        uint32_t* lbase = lbases[subtraction_index];
        int l = 0, r = size0[subtraction_index] - 1;
        uint32_t val = rbases[done1 % size1];
        //考虑之后换一下二分查找的写法，比如改为l < r，然后把mid的判断从循环里去掉，放到循环外(即最后l==r的时候)
        while (l <= r)
        {
            int mid = (l + r) >> 1;
            if (lbase[mid] < val)
                l = mid + 1;
            else if (lbase[mid] > val)
                r = mid - 1;
            else
            {
                atomicSub(&ret[subtraction_index], 1);
                break;
            }
        }
        done1 += THREADS_PER_WARP;
    }
    __threadfence_block();
    return ret;
}

__device__ int unordered_subtraction_size(const GPUVertexSet& set0, const GPUVertexSet& set1, int size_after_restrict = -1)
{
    return unordered_subtraction_size(set0.get_data_ptr(), set1.get_data_ptr(), set0.get_size(), set1.get_size(), size_after_restrict);
}

constexpr int MAX_SHARED_SET_LENGTH = 144; //如果一个集合小于这个阈值，则可以放在shared memory。需要与32x + 16对齐，为了两个subwarp同时做的时候没有bank conflict
__shared__ uint32_t local_mem[4 * MAX_SHARED_SET_LENGTH * WARPS_PER_BLOCK];

__device__ void global_to_shared(const GPUVertexSet& set, uint32_t* smem)
{
    int loop_size = set.get_size();
    int lid = threadIdx.x % THREADS_PER_WARP;
    for (int i = lid; i < loop_size; i += THREADS_PER_WARP)
        smem[i] = set.get_data(i); //频繁调用get_data，编译器应该会用寄存器缓存一下set.data_ptr吧？之后改一下手动存ptr试试
}

//将集合数据拷贝至shared memory，且每个元素值为原先的2倍（左移一位，最后一位作为flag位）
__device__ void global_to_shared_double(const GPUVertexSet& set, uint32_t* smem)
{
    int loop_size = set.get_size();
    int lid = threadIdx.x % THREADS_PER_WARP;
    for (int i = lid; i < loop_size; i += THREADS_PER_WARP)
        smem[i] = (set.get_data(i) << 1); //频繁调用get_data，编译器应该会用寄存器缓存一下set.data_ptr吧？之后改一下手动存ptr试试
}

//减少容斥原理中的计算量，并使用更多shared memory。但是性能却下降了
__device__ unsigned long long IEP_3_layer_more_shared(const GPUSchedule* schedule, GPUVertexSet* vertex_set, GPUVertexSet& subtraction_set,
    GPUVertexSet& tmp_set, int in_exclusion_optimize_num, int depth)
{
    //这个版本只需要用4个set大小的shared memory
    uint32_t* warp_mem_start = local_mem + 4 * MAX_SHARED_SET_LENGTH * (threadIdx.x / THREADS_PER_WARP);
    //首先找到需要做容斥原理的三个集合ABC的id
    int loop_set_prefix_ids0, loop_set_prefix_ids1, loop_set_prefix_ids2;
    loop_set_prefix_ids0 = schedule->get_loop_set_prefix_id(depth);
    loop_set_prefix_ids1 = schedule->get_loop_set_prefix_id(depth + 1);
    loop_set_prefix_ids2 = schedule->get_loop_set_prefix_id(depth + 2);
    //对3个集合从小到大排序
    if (vertex_set[loop_set_prefix_ids2].get_size() < vertex_set[loop_set_prefix_ids1].get_size())
        swap(loop_set_prefix_ids1, loop_set_prefix_ids2);
    if (vertex_set[loop_set_prefix_ids1].get_size() < vertex_set[loop_set_prefix_ids0].get_size())
        swap(loop_set_prefix_ids0, loop_set_prefix_ids1);
    if (vertex_set[loop_set_prefix_ids2].get_size() < vertex_set[loop_set_prefix_ids1].get_size())
        swap(loop_set_prefix_ids1, loop_set_prefix_ids2);

    uint32_t* subtraction_ptr = subtraction_set.get_data_ptr();
    int subtraction_size = subtraction_set.get_size();

    //把ABC从global移动到shared
    uint32_t *A_ptr, *B_ptr, *C_ptr;
    uint32_t A_size = vertex_set[loop_set_prefix_ids0].get_size(), B_size = vertex_set[loop_set_prefix_ids1].get_size(), C_size = vertex_set[loop_set_prefix_ids2].get_size();
    
    if (C_size < MAX_SHARED_SET_LENGTH) // 即ABC都可以被放入shared memory
    {
        //TODO： 其实ABC都能被放入shared memory这个限制太严了，实际上只要AB能放入shared memory就可以，也就是一开始只将AB的元素都*2，然后任何集合在与C求交的时候再处理一下就可以（如果C作为a数组，就取出C元素后*2，如果C作为b数组，就把a数组中的元素取出后除2）
        A_ptr = warp_mem_start;
        global_to_shared_double(vertex_set[loop_set_prefix_ids0], A_ptr);
        B_ptr = warp_mem_start + MAX_SHARED_SET_LENGTH;
        global_to_shared_double(vertex_set[loop_set_prefix_ids1], B_ptr);
        C_ptr = warp_mem_start + (MAX_SHARED_SET_LENGTH << 1);
        global_to_shared_double(vertex_set[loop_set_prefix_ids2], C_ptr);

        //首先对ABC求差集，就不用之后每次求完交集再次求差集了
        //real_X_size用于计算最终答案，X_size用于之后求交集（求差后X_size不变）
        //unsigned long long real_A_size = unordered_subtraction_with_flag(A_ptr, subtraction_ptr, A_size, subtraction_size);
        //unsigned long long real_B_size = unordered_subtraction_with_flag(B_ptr, subtraction_ptr, B_size, subtraction_size);
        //unsigned long long real_C_size = unordered_subtraction_with_flag(C_ptr, subtraction_ptr, C_size, subtraction_size);
        __shared__ uint32_t *lbases_block[MAX_SUBTRACTION_CNT * WARPS_PER_BLOCK];
        __shared__ int size0_block[MAX_SUBTRACTION_CNT * WARPS_PER_BLOCK];
        int wid = threadIdx.x / THREADS_PER_WARP;
        int lid = threadIdx.x % THREADS_PER_WARP;
        uint32_t **lbases = lbases_block + wid * MAX_SUBTRACTION_CNT;
        int *size0 = size0_block + wid * MAX_SUBTRACTION_CNT;
        if (lid == 0)
        {
            lbases[0] = A_ptr;
            lbases[1] = B_ptr;
            lbases[2] = C_ptr;
            size0[0] = A_size;
            size0[1] = B_size;
            size0[2] = C_size;
        }
        __threadfence_block();

        int* ret = unordered_subtraction_with_flag_parallel(lbases, subtraction_ptr, size0, subtraction_size, 3);
        unsigned long long real_A_size = ret[0];
        unsigned long long real_B_size = ret[1];
        unsigned long long real_C_size = ret[2];

        uint32_t* intersection_ptr = warp_mem_start + MAX_SHARED_SET_LENGTH * 3;
        //A & B
        unsigned long long AB_size = do_intersection_with_flag(intersection_ptr, A_ptr, B_ptr, A_size, B_size);
        //(A & B) & C
        unsigned long long ABC_size = get_intersection_size_with_flag(intersection_ptr, C_ptr, AB_size, C_size);
        //A & C
        unsigned long long AC_size = get_intersection_size_with_flag(A_ptr, C_ptr, A_size, C_size);
        //B & C
        unsigned long long BC_size = get_intersection_size_with_flag(B_ptr, C_ptr, B_size, C_size);
        return real_A_size * real_B_size * real_C_size - real_A_size * BC_size - real_B_size * AC_size - real_C_size * AB_size + (ABC_size << 1);
        //if (threadIdx.x == 0)
        //    printf("T %llu %llu %llu %llu %llu %llu %llu\n", real_A_size, real_B_size,real_C_size, AB_size, BC_size, AC_size, ABC_size);
    }
    else
    {
        if (A_size < MAX_SHARED_SET_LENGTH)
        {
            A_ptr = warp_mem_start;
            global_to_shared(vertex_set[loop_set_prefix_ids0], A_ptr);
        }
        else
            A_ptr = vertex_set[loop_set_prefix_ids0].get_data_ptr();
    
        if (B_size < MAX_SHARED_SET_LENGTH)
        {
            B_ptr = warp_mem_start + MAX_SHARED_SET_LENGTH;
            global_to_shared(vertex_set[loop_set_prefix_ids1], B_ptr);
        }
        else
            B_ptr = vertex_set[loop_set_prefix_ids1].get_data_ptr();
    
        if (C_size < MAX_SHARED_SET_LENGTH)
        {
            C_ptr = warp_mem_start + (MAX_SHARED_SET_LENGTH << 1);
            global_to_shared(vertex_set[loop_set_prefix_ids2], C_ptr);
        }
        else
            C_ptr = vertex_set[loop_set_prefix_ids2].get_data_ptr();
    
        //A & B，由于A.size < B.size，只要A.size < MAX_SHARED_SET_LENGTH，则求交后大小一定 < MAX_SHARED_SET_LENGTH，可以放到shared memory
        uint32_t* intersection_ptr = A_size < MAX_SHARED_SET_LENGTH ? (warp_mem_start + MAX_SHARED_SET_LENGTH * 3) : tmp_set.get_data_ptr();
        unsigned long long AB_size = do_intersection(intersection_ptr, A_ptr, B_ptr, A_size, B_size);
        AB_size = unordered_subtraction_size(intersection_ptr, subtraction_ptr, AB_size, subtraction_size);
        //(A & B) & C
        unsigned long long ABC_size = do_intersection(intersection_ptr, intersection_ptr, C_ptr, AB_size, C_size);
        ABC_size = unordered_subtraction_size(intersection_ptr, subtraction_ptr, ABC_size, subtraction_size);
        //A & C
        intersection_ptr = A_size < MAX_SHARED_SET_LENGTH ? (warp_mem_start + MAX_SHARED_SET_LENGTH * 3) : tmp_set.get_data_ptr();
        unsigned long long AC_size = do_intersection(intersection_ptr, A_ptr, C_ptr, A_size, C_size);
        AC_size = unordered_subtraction_size(intersection_ptr, subtraction_ptr, AC_size, subtraction_size);
        //B & C
        intersection_ptr = B_size < MAX_SHARED_SET_LENGTH ? (warp_mem_start + MAX_SHARED_SET_LENGTH * 3) : tmp_set.get_data_ptr();
        unsigned long long BC_size = do_intersection(intersection_ptr, B_ptr, C_ptr, B_size, C_size);
        /*BC_size = unordered_subtraction_size(intersection_ptr, subtraction_ptr, BC_size, subtraction_size);
    
        unsigned long long real_A_size = unordered_subtraction_size(A_ptr, subtraction_ptr, A_size, subtraction_size);
        unsigned long long real_B_size = unordered_subtraction_size(B_ptr, subtraction_ptr, B_size, subtraction_size);
        unsigned long long real_C_size = unordered_subtraction_size(C_ptr, subtraction_ptr, C_size, subtraction_size);*/

        __shared__ uint32_t *lbases_block[MAX_SUBTRACTION_CNT * WARPS_PER_BLOCK];
        __shared__ int size0_block[MAX_SUBTRACTION_CNT * WARPS_PER_BLOCK];
        int wid = threadIdx.x / THREADS_PER_WARP;
        int lid = threadIdx.x % THREADS_PER_WARP;
        uint32_t **lbases = lbases_block + wid * MAX_SUBTRACTION_CNT;
        int *size0 = size0_block + wid * MAX_SUBTRACTION_CNT;
        if (lid == 0)
        {
            lbases[0] = A_ptr;
            lbases[1] = B_ptr;
            lbases[2] = C_ptr;
            lbases[3] = intersection_ptr;
            size0[0] = A_size;
            size0[1] = B_size;
            size0[2] = C_size;
            size0[3] = BC_size;
        }
        __threadfence_block();

        int* ret = unordered_subtraction_parallel(lbases, subtraction_ptr, size0, subtraction_size, 4);
        unsigned long long real_A_size = ret[0];
        unsigned long long real_B_size = ret[1];
        unsigned long long real_C_size = ret[2];
        BC_size = ret[3];
        //if (threadIdx.x == 0)
        //    printf("T %llu %llu %llu %llu %llu %llu %llu\n", real_A_size, real_B_size,real_C_size, AB_size, BC_size, AC_size, ABC_size);
        return real_A_size * real_B_size * real_C_size - real_A_size * BC_size - real_B_size * AC_size - real_C_size * AB_size + (ABC_size << 1);
    }
}

//减少容斥原理中的计算量，并利用一定shared memory
__device__ unsigned long long IEP_3_layer(const GPUSchedule* schedule, GPUVertexSet* vertex_set, GPUVertexSet& subtraction_set,
    GPUVertexSet& tmp_set, int in_exclusion_optimize_num, int depth)
{
    //这个版本只需要用一个set大小的shared memory
    uint32_t* warp_mem_start = local_mem + MAX_SHARED_SET_LENGTH * (threadIdx.x / THREADS_PER_WARP);
    //首先找到需要做容斥原理的三个集合ABC的id
    int loop_set_prefix_ids0, loop_set_prefix_ids1, loop_set_prefix_ids2;
    loop_set_prefix_ids0 = schedule->get_loop_set_prefix_id(depth);
    loop_set_prefix_ids1 = schedule->get_loop_set_prefix_id(depth + 1);
    loop_set_prefix_ids2 = schedule->get_loop_set_prefix_id(depth + 2);
    //对3个集合从小到大排序
    if (vertex_set[loop_set_prefix_ids2].get_size() < vertex_set[loop_set_prefix_ids1].get_size())
        swap(loop_set_prefix_ids1, loop_set_prefix_ids2);
    if (vertex_set[loop_set_prefix_ids1].get_size() < vertex_set[loop_set_prefix_ids0].get_size())
        swap(loop_set_prefix_ids0, loop_set_prefix_ids1);
    if (vertex_set[loop_set_prefix_ids2].get_size() < vertex_set[loop_set_prefix_ids1].get_size())
        swap(loop_set_prefix_ids1, loop_set_prefix_ids2);

    uint32_t* subtraction_ptr = subtraction_set.get_data_ptr();
    int subtraction_size = subtraction_set.get_size();
    //A & B，由于A.size < B.size，只要A.size < MAX_SHARED_SET_LENGTH，则求交后大小一定 < MAX_SHARED_SET_LENGTH，可以放到shared memory
    uint32_t* intersection_ptr = vertex_set[loop_set_prefix_ids0].get_size() < MAX_SHARED_SET_LENGTH ? warp_mem_start : tmp_set.get_data_ptr();
    unsigned long long AB_size = do_intersection(intersection_ptr, vertex_set[loop_set_prefix_ids0].get_data_ptr(), vertex_set[loop_set_prefix_ids1].get_data_ptr(), vertex_set[loop_set_prefix_ids0].get_size(), vertex_set[loop_set_prefix_ids1].get_size());
    AB_size = unordered_subtraction_size(intersection_ptr, subtraction_ptr, AB_size, subtraction_size);
    //(A & B) & C
    unsigned long long ABC_size = do_intersection(intersection_ptr, intersection_ptr, vertex_set[loop_set_prefix_ids2].get_data_ptr(), AB_size, vertex_set[loop_set_prefix_ids2].get_size());
    ABC_size = unordered_subtraction_size(intersection_ptr, subtraction_ptr, ABC_size, subtraction_size);
    //A & C
    intersection_ptr = vertex_set[loop_set_prefix_ids0].get_size() < MAX_SHARED_SET_LENGTH ? warp_mem_start : tmp_set.get_data_ptr();
    unsigned long long AC_size = do_intersection(intersection_ptr, vertex_set[loop_set_prefix_ids0].get_data_ptr(), vertex_set[loop_set_prefix_ids2].get_data_ptr(), vertex_set[loop_set_prefix_ids0].get_size(), vertex_set[loop_set_prefix_ids2].get_size());
    AC_size = unordered_subtraction_size(intersection_ptr, subtraction_ptr, AC_size, subtraction_size);
    //B & C
    intersection_ptr = vertex_set[loop_set_prefix_ids1].get_size() < MAX_SHARED_SET_LENGTH ? warp_mem_start : tmp_set.get_data_ptr();
    unsigned long long BC_size = do_intersection(intersection_ptr, vertex_set[loop_set_prefix_ids1].get_data_ptr(), vertex_set[loop_set_prefix_ids2].get_data_ptr(), vertex_set[loop_set_prefix_ids1].get_size(), vertex_set[loop_set_prefix_ids2].get_size());
    BC_size = unordered_subtraction_size(intersection_ptr, subtraction_ptr, BC_size, subtraction_size);

    unsigned long long A_size = unordered_subtraction_size(vertex_set[loop_set_prefix_ids0], subtraction_set);
    unsigned long long B_size = unordered_subtraction_size(vertex_set[loop_set_prefix_ids1], subtraction_set);
    unsigned long long C_size = unordered_subtraction_size(vertex_set[loop_set_prefix_ids2], subtraction_set);
    return A_size * B_size * C_size - A_size * BC_size - B_size * AC_size - C_size * AB_size + (ABC_size << 1);
}

/**
 * @brief 最终层的容斥原理优化计算。
 */
__device__ void GPU_pattern_matching_final_in_exclusion(const GPUSchedule* schedule, GPUVertexSet* vertex_set, GPUVertexSet& subtraction_set,
    GPUVertexSet& tmp_set, unsigned long long& local_ans, int depth, uint32_t *edge, uint32_t *vertex)
{
    int in_exclusion_optimize_num = schedule->get_in_exclusion_optimize_num();
    if (in_exclusion_optimize_num == 3) {
        local_ans += IEP_3_layer_more_shared(schedule, vertex_set, subtraction_set, tmp_set, in_exclusion_optimize_num, depth);
        return;
    }
    //int* loop_set_prefix_ids[ in_exclusion_optimize_num ];
    __shared__ int loop_set_prefix_ids_block[8 * WARPS_PER_BLOCK];//偷懒用了static，之后考虑改成dynamic
    int* loop_set_prefix_ids = loop_set_prefix_ids_block + threadIdx.x / THREADS_PER_WARP * 8;
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
                int tmp = unordered_subtraction_size(vertex_set[id], subtraction_set);
                val = val * tmp;
            } else {
                int id = loop_set_prefix_ids[cur_graph.data[cur_graph_rank].data[0]];
                tmp_set.copy_from(vertex_set[id]);

                for (int i = 1; i < cur_graph.data[cur_graph_rank].size; ++i) {
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
}

constexpr int MAX_DEPTH = 5; // 非递归pattern matching支持的最大深度

template <int depth>
__device__ void GPU_pattern_matching_func(const GPUSchedule* schedule, GPUVertexSet* vertex_set, GPUVertexSet& subtraction_set,
    GPUVertexSet& tmp_set, unsigned long long& local_ans, uint32_t *edge, uint32_t *vertex)
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
__device__ void GPU_pattern_matching_func<MAX_DEPTH>(const GPUSchedule* schedule, GPUVertexSet* vertex_set, GPUVertexSet& subtraction_set,
    GPUVertexSet& tmp_set, unsigned long long& local_ans, uint32_t *edge, uint32_t *vertex)
{
    // assert(false);
}

__global__ void gpu_pattern_matching(uint32_t edge_num, uint32_t buffer_size, uint32_t *edge_from, uint32_t *edge, uint32_t *vertex, uint32_t *tmp, const GPUSchedule* schedule) {
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
    GPUVertexSet& subtraction_set = vertex_set[num_prefixes];
    GPUVertexSet& tmp_set = vertex_set[num_prefixes + 1];

    __threadfence_block(); //之后考虑把所有的syncthreads都改成syncwarp


    uint32_t v0, v1;
    uint32_t l, r;

    unsigned long long sum = 0;

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
        
        unsigned long long local_sum = 0; // local sum (corresponding to an edge index)
        GPU_pattern_matching_func<2>(schedule, vertex_set, subtraction_set, tmp_set, local_sum, edge, vertex);
        // GPU_pattern_matching_aggressive_func(schedule, vertex_set, subtraction_set, tmp_set, local_sum, 2, edge, vertex);
        sum += local_sum;
    }

    if (lid == 0) {
        atomicAdd(&dev_sum, sum);
    }
}

void pattern_matching_init(Graph *g, const Schedule& schedule) {
    schedule.print_schedule();
    uint32_t *edge_from = new uint32_t[g->e_cnt];
    for(uint32_t i = 0; i < g->v_cnt; ++i)
        for(uint32_t j = g->vertex[i]; j < g->vertex[i+1]; ++j)
            edge_from[j] = i;

    uint32_t *edge = new uint32_t[g->e_cnt];
    uint32_t *vertex = new uint32_t[g->v_cnt + 1];

    for(uint32_t i = 0;i < g->e_cnt; ++i) edge[i] = g->edge[i];
    for(uint32_t i = 0;i <= g->v_cnt; ++i) vertex[i] = g->vertex[i];

    tmpTime.check(); 

    int num_blocks = 4096;
    int num_total_warps = num_blocks * WARPS_PER_BLOCK;

    uint32_t size_edge = g->e_cnt * sizeof(uint32_t);
    uint32_t size_vertex = (g->v_cnt + 1) * sizeof(uint32_t);
    uint32_t size_tmp = VertexSet::max_intersection_size * sizeof(uint32_t) * num_total_warps * (schedule.get_total_prefix_num() + 2); //prefix + subtraction + tmp

    uint32_t *dev_edge;
    uint32_t *dev_edge_from;
    uint32_t *dev_vertex;
    uint32_t *dev_tmp;

    gpuErrchk( cudaMalloc((void**)&dev_edge, size_edge));
    gpuErrchk( cudaMalloc((void**)&dev_edge_from, size_edge));
    gpuErrchk( cudaMalloc((void**)&dev_vertex, size_vertex));
    gpuErrchk( cudaMalloc((void**)&dev_tmp, size_tmp));

    gpuErrchk( cudaMemcpy(dev_edge, edge, size_edge, cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy(dev_edge_from, edge_from, size_edge, cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy(dev_vertex, vertex, size_vertex, cudaMemcpyHostToDevice));

    unsigned long long sum = 0;

    //memcpy schedule
    GPUSchedule* dev_schedule;
    gpuErrchk( cudaMallocManaged((void**)&dev_schedule, sizeof(GPUSchedule)));
    dev_schedule->transform_in_exclusion_optimize_group_val(schedule);
    int schedule_size = schedule.get_size();
    int max_prefix_num = schedule_size * (schedule_size - 1) / 2;
    gpuErrchk( cudaMallocManaged((void**)&dev_schedule->adj_mat, sizeof(int) * schedule_size * schedule_size));
    gpuErrchk( cudaMemcpy(dev_schedule->adj_mat, schedule.get_adj_mat_ptr(), sizeof(int) * schedule_size * schedule_size, cudaMemcpyHostToDevice));

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

    dev_schedule->size = schedule.get_size();
    dev_schedule->total_prefix_num = schedule.get_total_prefix_num();
    dev_schedule->total_restrict_num = schedule.get_total_restrict_num();
    dev_schedule->in_exclusion_optimize_num = schedule.get_in_exclusion_optimize_num();
    dev_schedule->k_val = schedule.get_k_val();

    printf("schedule.prefix_num: %d\n", schedule.get_total_prefix_num());
    printf("shared memory for vertex set per block: %ld bytes\n", 
        (schedule.get_total_prefix_num() + 2) * WARPS_PER_BLOCK * sizeof(GPUVertexSet));

    tmpTime.print("Prepare time cost");
    tmpTime.check();

    uint32_t edge_num = g->e_cnt;
    uint32_t buffer_size = VertexSet::max_intersection_size;
    uint32_t block_shmem_size = (schedule.get_total_prefix_num() + 2) * WARPS_PER_BLOCK * sizeof(GPUVertexSet);

    int device;
    cudaDeviceProp prop;
    int activeWarps;
    int maxWarps;
    int numBlocks;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &numBlocks,
    gpu_pattern_matching,
    WARPS_PER_BLOCK,
    0);
    activeWarps = numBlocks * THREADS_PER_BLOCK / prop.warpSize;
    maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;
    printf("Occupancy: numBlocks = %d THREADS_PER_BLOCK = %d activeWarp = %d maxWarps = %d\n", numBlocks, THREADS_PER_BLOCK, activeWarps, maxWarps);
    // 注意：此处没有错误，buffer_size代指每个顶点集所需的int数目，无需再乘sizeof(uint32_t)，但是否考虑对齐？
    //因为目前用了managed开内存，所以第一次运行kernel会有一定额外开销，考虑运行两次，第一次作为warmup
    gpu_pattern_matching<<<num_blocks, THREADS_PER_BLOCK, block_shmem_size>>>
        (edge_num, buffer_size, dev_edge_from, dev_edge, dev_vertex, dev_tmp, dev_schedule);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk( cudaMemcpyFromSymbol(&sum, dev_sum, sizeof(sum)) );

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
    gpuErrchk(cudaFree(dev_schedule));

    delete[] edge, edge_from, vertex;
}

int main(int argc,char *argv[]) {
    Graph *g;
    DataLoader D;

    if (argc < 3) {
        printf("Example Usage: %s Patents ~zms/patents_input\n", argv[0]);
        return 0;
    }

    const std::string type = argv[1];
    const std::string path = argv[2];

    DataType my_type;

    GetDataType(my_type, type);

    if(my_type == DataType::Invalid) {
        printf("Dataset not found!\n");
        return 0;
    }

    using std::chrono::system_clock;
    auto t1 = system_clock::now();

    bool ok = D.load_data(g, my_type, path.c_str());
    //todo: check ok

    auto t2 = system_clock::now();
    auto load_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    printf("Load data success! time: %g seconds\n", load_time.count() / 1.0e6);
    fflush(stdout);

    allTime.check();

    //const char *pattern_str = "0111010011100011100001100"; // 5 house p1
    const char *pattern_str = "011011101110110101011000110000101000"; // 6 p2
    // const char *pattern_str = "0111111101111111011101110100111100011100001100000"; // 7 p5

    Pattern p(6, pattern_str);
    printf("pattern = \n");
    p.print();
    printf("max intersection size %d\n", VertexSet::max_intersection_size);
    bool is_pattern_valid;
    bool use_in_exclusion_optimize = true;
    Schedule schedule(p, is_pattern_valid, 1, 1, use_in_exclusion_optimize, g->v_cnt, g->e_cnt, g->tri_cnt);
    //Schedule schedule(p, is_pattern_valid, 0, 1, use_in_exclusion_optimize, g->v_cnt, g->e_cnt, g->tri_cnt); // use the best schedule
    //todo : check is_pattern_valid

    pattern_matching_init(g, schedule);

    allTime.print("Total time cost");

    return 0;
}
