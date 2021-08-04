/**
 * 这个版本里面没有细粒度计时。有计时的在gpu_graph_with_timer.cu里面。
 * 而且计时的方式与zms版本略有区别。
 */
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

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <sys/time.h>
#include <chrono>
#include <unistd.h>

#define likely(x)   __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
constexpr int THREADS_PER_BLOCK = 256;
constexpr int THREADS_PER_WARP = 32;
constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / THREADS_PER_WARP;

int stdout_fd;

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
    /*
    __host__ void transform_in_exclusion_optimize_group_val(const Schedule& schedule)
    {
        // @todo 注意当容斥优化无法使用时，内存分配会失败。需要修正 
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
    */

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

// __global__ void gen_final(const GPUSchedule* schedule) {
//     if(threadIdx.x == 0) {
//         printf("__device__ void GPU_pattern_matching_final_in_exclusion(GPUVertexSet* vertex_set, GPUVertexSet& subtraction_set,GPUVertexSet& tmp_set, unsigned long long& local_ans,  uint32_t *edge, uint32_t *vertex) {\n");
//         printf("extern __shared__ char ans_array[];\n");
//         printf("int* ans = ((int*) (ans_array + %d)) + %d * (threadIdx.x / THREADS_PER_WARP);\n", schedule->ans_array_offset, schedule->in_exclusion_optimize_vertex_id_size);
//         for(int i = 0; i < schedule->in_exclusion_optimize_vertex_id_size; ++i) {
//             if(schedule->in_exclusion_optimize_vertex_flag[i]) {
//                 printf("ans[%d] = vertex_set[%d].get_size() - %d;\n", i, schedule->in_exclusion_optimize_vertex_id[i], schedule->in_exclusion_optimize_vertex_coef[i]);
//             }
//             else {
//                 printf("ans[%d] = unordered_subtraction_size(vertex_set[%d], subtraction_set);\n", i, schedule->in_exclusion_optimize_vertex_id[i]);
//             }
//         }
//         int last_pos = -1;
//         printf("long long val;\n");
//         for(int pos = 0; pos < schedule->in_exclusion_optimize_array_size; ++pos) {
//             if(pos == last_pos + 1) {
//                 printf("val = ans[%d];\n", schedule->in_exclusion_optimize_ans_pos[pos]);
//             }
//             else {
//                 printf("val = val * ans[%d];\n", schedule->in_exclusion_optimize_ans_pos[pos]);
//             }
//             if(schedule->in_exclusion_optimize_flag[pos]) {
//                 last_pos = pos;
//                 printf("local_ans += val * %d;\n", schedule->in_exclusion_optimize_coef[pos]);
//             }
//         }
//         printf("}\n");
//     }
// }

#define write_code(fmt, ...) do { \
    for (int _ = 0; _ < indentation; ++_) printf(" "); \
    printf(fmt, ## __VA_ARGS__); \
} while (0)

#define write_code_noindent printf

__device__ void gen_build_vertex_set(const GPUSchedule* schedule, int prefix_id, int indentation, const char* input_data_str, const char* input_size_str)
{
    int father_id = schedule->get_father_prefix_id(prefix_id);
    if (father_id == -1)
    {
        write_code("if (threadIdx.x % THREADS_PER_WARP == 0)\n");
        write_code("    vertex_set[%d].init(%s, %s);\n", prefix_id, input_size_str, input_data_str);
        write_code("__threadfence_block();\n");
    }
    else
    {
        bool only_need_size = schedule->only_need_size[prefix_id];
        static bool first_time = true;
        if (first_time)
            write_code("GPUVertexSet* tmp_vset;\n");
        first_time = false;
        if(only_need_size) {
            write_code("{\n");
            indentation += 4;
            write_code("tmp_vset = &vertex_set[%d];\n", prefix_id);
            write_code("if (threadIdx.x % THREADS_PER_WARP == 0)\n");
            write_code("    tmp_vset->init(%s, %s);\n", input_size_str, input_data_str);
            write_code("__threadfence_block();\n");
            write_code("if (%s > vertex_set[%d].get_size())\n", input_size_str, father_id);
            write_code("    tmp_vset->size -= unordered_subtraction_size(*tmp_vset, vertex_set[%d], -1);\n", father_id);
            write_code("else\n");
            write_code("    tmp_vset->size = vertex_set[%d].get_size() - unordered_subtraction_size(vertex_set[%d], *tmp_vset, -1);\n", father_id, father_id);
            indentation -= 4;
            write_code("}\n");
        }
        else {
            write_code("intersection2(vertex_set[%d].get_data_ptr(), vertex_set[%d].get_data_ptr(), %s, vertex_set[%d].get_size(), %s, &vertex_set[%d].size);\n", prefix_id, father_id, input_data_str, father_id, input_size_str, prefix_id);
        }
    }
}


__global__ void gen_GPU_pattern_matching_func(const GPUSchedule* schedule)
{
    if(threadIdx.x == 0) {
        //WORK SPACE BEGIN
        int indentation = 0;
        //如果图也能确定的话，edge_num也可以确定
        write_code("__global__ void gpu_pattern_matching(uint32_t edge_num, uint32_t buffer_size, uint32_t *edge_from, uint32_t *edge, uint32_t *vertex, uint32_t *tmp, const GPUSchedule* schedule) {\n");
        indentation += 4;
        write_code("__shared__ unsigned int block_edge_idx[WARPS_PER_BLOCK];\n");
        write_code("extern __shared__ GPUVertexSet block_vertex_set[];\n");
        write_code("extern __shared__ char block_shmem[];\n\n");

        int num_prefixes = schedule->get_total_prefix_num();
        int num_vertex_sets_per_warp = num_prefixes + 1;

        write_code("int wid = threadIdx.x / THREADS_PER_WARP;\n");
        write_code("int lid = threadIdx.x % THREADS_PER_WARP;\n");
        write_code("int global_wid = blockIdx.x * WARPS_PER_BLOCK + wid;\n");
        write_code("unsigned int &edge_idx = block_edge_idx[wid];\n");
        write_code("GPUVertexSet *vertex_set = block_vertex_set + wid * %d;\n\n", num_vertex_sets_per_warp);

        write_code("GPUVertexSet &subtraction_set = vertex_set[%d];\n", num_prefixes);
        // write_code("GPUVertexSet &tmp_set = vertex_set[%d];\n", num_prefixes + 1);

        write_code("if (lid == 0) {\n");
        indentation += 4;
        write_code("edge_idx = 0;\n");
        write_code("uint32_t offset = buffer_size * global_wid * %d;\n\n", num_prefixes);
        
        write_code("uint32_t *block_subtraction_set_buf = (uint32_t *)(block_shmem + %d);\n", schedule->ans_array_offset);
        write_code("subtraction_set.set_data_ptr(block_subtraction_set_buf + wid * %d);\n\n", schedule->get_size() - schedule->get_in_exclusion_optimize_num());

        write_code("for (int i = 0; i < %d; ++i) {\n", num_prefixes);
        indentation += 4;
        write_code("vertex_set[i].set_data_ptr(tmp + offset);\n");
        write_code("offset += buffer_size;\n");
        indentation -= 4;
        write_code("}\n");
        indentation -= 4;
        write_code("}\n\n");

        write_code("__threadfence_block();\n\n");

        write_code("uint32_t v0, v1");
        // 直接以局部变量代替原先的subtraction_set
        for (int depth = 2; depth < schedule->get_size() - schedule->get_in_exclusion_optimize_num(); ++depth)
            write_code_noindent(", v%d", depth);
        write_code_noindent(";\n");
        write_code("uint32_t l, r;\n");

        write_code("unsigned long long sum = 0;\n\n");

        write_code("while (true) {\n");
        indentation += 4;
        write_code("if (lid == 0) {\n");
        indentation += 4;
        write_code("edge_idx = atomicAdd(&dev_cur_edge, 1);\n");
        // write_code("unsigned int i = edge_idx;\n");
        // write_code("if (i < edge_num) {\n");
        // indentation += 4;
        // write_code("subtraction_set.init();\n");
        // write_code("subtraction_set.push_back(edge_from[i]);\n");
        // write_code("subtraction_set.push_back(edge[i]);\n");
        // indentation -= 4;
        // write_code("}\n");
        indentation -= 4;
        write_code("}\n");

        write_code("__threadfence_block();\n\n");

        write_code("unsigned int i = edge_idx;\n");
        write_code("if (i >= edge_num) break;\n\n");
       
        write_code("v0 = edge_from[i];\n");
        write_code("v1 = edge[i];\n");

        if (schedule->get_restrict_last(1) != -1) {
            write_code("if (v0 <= v1) continue;\n\n");
        } 

        //write_code("bool is_zero = false;\n");
        write_code("get_edge_index(v0, l, r);\n");
        for (int prefix_id = schedule->get_last(0); prefix_id != -1; prefix_id = schedule->get_next(prefix_id)) {
            gen_build_vertex_set(schedule, prefix_id, indentation, "&edge[l]", "r - l");
            write_code("\n");
        }

        write_code("get_edge_index(v1, l, r);\n");
        for (int prefix_id = schedule->get_last(1); prefix_id != -1; prefix_id = schedule->get_next(prefix_id))
        {
            //write_code("vertex_set[%d].build_vertex_set(%d, vertex_set, &edge[l], r - l);\n", prefix_id, schedule->get_father_prefix_id(prefix_id));
            gen_build_vertex_set(schedule, prefix_id, indentation, "&edge[l]", "r - l");
            if (prefix_id < schedule->get_basic_prefix_num())
                write_code("if (vertex_set[%d].get_size() == 0) continue;\n", prefix_id); //因为代码生成后没有外层的for循环了，所以不需要先break再continue了
            write_code("\n");    
        }
        //TODO
        //TODO
        //TODO
        
        // write_code("extern __shared__ char ans_array[];\n");
        // write_code("int* ans = ((int*) (ans_array + %d)) + %d * (threadIdx.x / THREADS_PER_WARP);\n\n", schedule->ans_array_offset, schedule->in_exclusion_optimize_vertex_id_size);
        
        for(int depth = 2; depth < schedule->get_size() - schedule->get_in_exclusion_optimize_num();  ++depth) {
            int loop_set_prefix_id = schedule->get_loop_set_prefix_id(depth);
            write_code("int loop_size_depth%d = vertex_set[%d].get_size();\n", depth, loop_set_prefix_id);
            // write_code("if (loop_size_depth%d <= 0) continue;\n", depth);

            write_code("uint32_t* loop_data_ptr_depth%d = vertex_set[%d].get_data_ptr();\n",depth, loop_set_prefix_id);
            
            bool has_min_vertex = schedule->get_restrict_last(depth) != -1;
            if(has_min_vertex) {
                write_code("uint32_t min_vertex_depth%d = 0xffffffff;\n", depth);
            }

            for (int i = schedule->get_restrict_last(depth); i != -1; i = schedule->get_restrict_next(i)) {
                // write_code("if(min_vertex_depth%d > subtraction_set.get_data(%d)) min_vertex_depth%d = subtraction_set.get_data(%d);\n", depth, schedule->get_restrict_index(i), depth, schedule->get_restrict_index(i));
                write_code("min_vertex_depth%d = min(min_vertex_depth%d, v%d);\n", depth, depth, schedule->get_restrict_index(i));
            }

            write_code("for (int i_depth%d = 0; i_depth%d < loop_size_depth%d; ++i_depth%d) {\n", depth, depth, depth, depth);
            indentation += 4;
            write_code("uint32_t v_depth%d = loop_data_ptr_depth%d[i_depth%d];\n", depth, depth, depth);
            if(has_min_vertex) {
                write_code("if (min_vertex_depth%d <= v_depth%d) break;\n", depth, depth);
            }

            // write_code("if (subtraction_set.has_data(v_depth%d)) continue;\n", depth);
            write_code("if (v0 == v_depth%d || v1 == v_depth%d", depth, depth);
            for (int i = 2; i < depth; ++i)
                write_code_noindent(" || v%d == v_depth%d", i, depth);
            write_code_noindent(") continue;\n\n");

            write_code("unsigned int l_depth%d, r_depth%d;\n", depth, depth);
            write_code("get_edge_index(v_depth%d, l_depth%d, r_depth%d);\n", depth, depth, depth);

            for (int prefix_id = schedule->get_last(depth); prefix_id != -1; prefix_id = schedule->get_next(prefix_id)) {
                char data_str[64] = "&edge[l_depth?]";
                data_str[13] = depth + '0';
                char size_str[64] = "r_depth? - l_depth?";
                size_str[7] = depth + '0';
                size_str[18] = depth + '0';
                //sprintf(data_str, "&edge[l_depth%d]", depth);
                //sprintf(size_str, "r_depth%d - l_depth%d", depth, depth);
                gen_build_vertex_set(schedule, prefix_id, indentation, data_str, size_str);
                if (prefix_id < schedule->get_basic_prefix_num()) //在general版本中没有这个判断，是因为general加了这个判断会更慢（不如直接判断break_size），但在这个版本加了这个判断可能会省去之后的break_size判断
                    write_code("if (vertex_set[%d].get_size() == %d) continue;\n", prefix_id, schedule->get_break_size(prefix_id));
                write_code("\n");
            }

            // write_code("if (threadIdx.x % THREADS_PER_WARP == 0) subtraction_set.push_back(v_depth%d);\n", depth);
            // write_code("__threadfence_block();\n\n");
            write_code("v%d = v_depth%d; // subtraction_set.push_back(v%d);\n\n", depth, depth, depth);
        }
        
        bool subtraction_set_required = false;
        for (int i = 0; i < schedule->in_exclusion_optimize_vertex_id_size; ++i) {
            if (!schedule->in_exclusion_optimize_vertex_flag[i]) {
                subtraction_set_required = true;
                break;
            }
        }
        if (subtraction_set_required) {
            // build subtraction set
            write_code("if (lid == 0) {\n");
            indentation += 4;
            write_code("uint32_t *p = subtraction_set.get_data_ptr();\n");
            for (int i = 0; i < schedule->get_size() - schedule->get_in_exclusion_optimize_num(); ++i)
                write_code("p[%d] = v%d;\n", i, i);
            write_code("subtraction_set.set_size(%d);\n", schedule->get_size() - schedule->get_in_exclusion_optimize_num());
            indentation -= 4;
            write_code("}\n");
            write_code("__threadfence_block();\n\n");
        }

        for(int i = 0; i < schedule->in_exclusion_optimize_vertex_id_size; ++i) {
            if(schedule->in_exclusion_optimize_vertex_flag[i]) {
                write_code("int ans%d = vertex_set[%d].get_size() - %d;\n", i, schedule->in_exclusion_optimize_vertex_id[i], schedule->in_exclusion_optimize_vertex_coef[i]);
            }
            else {
                write_code("int ans%d = unordered_subtraction_size(vertex_set[%d], subtraction_set);\n", i, schedule->in_exclusion_optimize_vertex_id[i]);
            }
        }
        int last_pos = -1;
        write_code("long long val;\n");
        for(int pos = 0; pos < schedule->in_exclusion_optimize_array_size; ++pos) {
            if(pos == last_pos + 1) {
                write_code("val = ans%d;\n", schedule->in_exclusion_optimize_ans_pos[pos]);
            }
            else {
                write_code("val = val * ans%d;\n", schedule->in_exclusion_optimize_ans_pos[pos]);
            }
            if(schedule->in_exclusion_optimize_flag[pos]) {
                last_pos = pos;
                write_code("sum += val * %d;\n", schedule->in_exclusion_optimize_coef[pos]);
            }
        }

        for(int depth = schedule->get_size() - schedule->get_in_exclusion_optimize_num() - 1; depth >= 2; --depth) {
            // write_code("if (threadIdx.x % THREADS_PER_WARP == 0) subtraction_set.pop_back();\n");
            // write_code("__threadfence_block();\n");

            indentation -= 4;
            write_code("}\n");
        }
        indentation -= 4;
        write_code("}\n");
        write_code("if (lid == 0) atomicAdd(&dev_sum, sum);\n");
        indentation -= 4;
        write_code("}\n");
    }
}

void pattern_matching_init(Graph *g, const Schedule_IEP& schedule_iep) {
    //printf("basic prefix %d, total prefix %d\n", schedule_iep.get_basic_prefix_num(), schedule_iep.get_total_prefix_num());

    int num_blocks = 1024;
    int num_total_warps = num_blocks * WARPS_PER_BLOCK;

    size_t size_edge = g->e_cnt * sizeof(uint32_t);
    size_t size_vertex = (g->v_cnt + 1) * sizeof(uint32_t);
    // size_t size_tmp = VertexSet::max_intersection_size * sizeof(uint32_t) * num_total_warps * (schedule_iep.get_total_prefix_num() + 2); //prefix + subtraction + tmp
    size_t size_tmp = VertexSet::max_intersection_size * sizeof(uint32_t) * num_total_warps * schedule_iep.get_total_prefix_num();
    
    //schedule_iep.print_schedule();
    uint32_t *edge_from = new uint32_t[g->e_cnt];
    for(uint32_t i = 0; i < g->v_cnt; ++i)
        for(uint32_t j = g->vertex[i]; j < g->vertex[i+1]; ++j)
            edge_from[j] = i;

    tmpTime.check(); 

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

    //printf("array size %d\n", in_exclusion_optimize_array_size);
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

    gpuErrchk( cudaMallocManaged((void**)&dev_schedule->in_exclusion_optimize_vertex_id, sizeof(int) * in_exclusion_optimize_vertex_id_size));
    gpuErrchk( cudaMemcpy(dev_schedule->in_exclusion_optimize_vertex_id, in_exclusion_optimize_vertex_id, sizeof(int) * in_exclusion_optimize_vertex_id_size, cudaMemcpyHostToDevice));
    
    gpuErrchk( cudaMallocManaged((void**)&dev_schedule->in_exclusion_optimize_vertex_flag, sizeof(bool) * in_exclusion_optimize_vertex_id_size));
    gpuErrchk( cudaMemcpy(dev_schedule->in_exclusion_optimize_vertex_flag, in_exclusion_optimize_vertex_flag, sizeof(bool) * in_exclusion_optimize_vertex_id_size, cudaMemcpyHostToDevice));
    
    gpuErrchk( cudaMallocManaged((void**)&dev_schedule->in_exclusion_optimize_vertex_coef, sizeof(int) * in_exclusion_optimize_vertex_id_size));
    gpuErrchk( cudaMemcpy(dev_schedule->in_exclusion_optimize_vertex_coef, in_exclusion_optimize_vertex_coef, sizeof(int) * in_exclusion_optimize_vertex_id_size, cudaMemcpyHostToDevice));

    gpuErrchk( cudaMallocManaged((void**)&dev_schedule->in_exclusion_optimize_coef, sizeof(int) * in_exclusion_optimize_array_size));
    gpuErrchk( cudaMemcpy(dev_schedule->in_exclusion_optimize_coef, in_exclusion_optimize_coef, sizeof(int) * in_exclusion_optimize_array_size, cudaMemcpyHostToDevice));

    gpuErrchk( cudaMallocManaged((void**)&dev_schedule->in_exclusion_optimize_flag, sizeof(bool) * in_exclusion_optimize_array_size));
    gpuErrchk( cudaMemcpy(dev_schedule->in_exclusion_optimize_flag, in_exclusion_optimize_flag, sizeof(bool) * in_exclusion_optimize_array_size, cudaMemcpyHostToDevice));
    
    gpuErrchk( cudaMallocManaged((void**)&dev_schedule->in_exclusion_optimize_ans_pos, sizeof(int) * in_exclusion_optimize_array_size));
    gpuErrchk( cudaMemcpy(dev_schedule->in_exclusion_optimize_ans_pos, in_exclusion_optimize_ans_pos, sizeof(int) * in_exclusion_optimize_array_size, cudaMemcpyHostToDevice));

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

    //printf("schedule_iep.prefix_num: %d\n", schedule_iep.get_total_prefix_num());
    //printf("shared memory for vertex set per block: %ld bytes\n", 
    //    (schedule_iep.get_total_prefix_num() + 2) * WARPS_PER_BLOCK * sizeof(GPUVertexSet) + in_exclusion_optimize_vertex_id_size * WARPS_PER_BLOCK * sizeof(int));

    //tmpTime.print("Prepare time cost");
    tmpTime.check();
 
    uint32_t buffer_size = VertexSet::max_intersection_size; // 注意：此处没有错误，buffer_size代指每个顶点集所需的int数目，无需再乘sizeof(uint32_t)，但是否考虑对齐？
    uint32_t block_subtraction_set_size = (schedule_iep.get_size() - schedule_iep.get_in_exclusion_optimize_num()) * WARPS_PER_BLOCK * sizeof(uint32_t);
    uint32_t block_shmem_size = (schedule_iep.get_total_prefix_num() + 1) * WARPS_PER_BLOCK * sizeof(GPUVertexSet) + block_subtraction_set_size;
    dev_schedule->ans_array_offset = block_shmem_size - block_subtraction_set_size;
    // printf("block_shmem: %u subtraction reserve: %d offset: %u\n", block_shmem_size, block_subtraction_set_size, dev_schedule->ans_array_offset);
    // ans_array_offset的意义已改变，是block内subtraction_set实际空间的偏移（以字节计）
     
    // uint32_t block_shmem_size = (schedule_iep.get_total_prefix_num() + 2) * WARPS_PER_BLOCK * sizeof(GPUVertexSet) + in_exclusion_optimize_vertex_id_size * WARPS_PER_BLOCK * sizeof(int);
    // dev_schedule->ans_array_offset = block_shmem_size - in_exclusion_optimize_vertex_id_size * WARPS_PER_BLOCK * sizeof(int);

    //int max_active_blocks_per_sm;
    //cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks_per_sm, gpu_pattern_matching, THREADS_PER_BLOCK, block_shmem_size);
    //printf("max number of active warps per SM: %d\n", max_active_blocks_per_sm * WARPS_PER_BLOCK);
    
    //gpu_pattern_matching<<<num_blocks, THREADS_PER_BLOCK, block_shmem_size>>>
      //  (g->e_cnt, buffer_size, dev_edge_from, dev_edge, dev_vertex, dev_tmp, dev_schedule);
    fflush(stdout);
    //fclose(stdout);
    dup2(stdout_fd, STDOUT_FILENO); //之前重定向到文件主要是为了把一些初始化的输出重定向，现在重定向回控制台

    gen_GPU_pattern_matching_func<<<1,1>>>(dev_schedule);
    fflush(stdout);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk( cudaMemcpyFromSymbol(&sum, dev_sum, sizeof(sum)) );

    sum /= schedule_iep.get_in_exclusion_optimize_redundancy();
    
    #ifdef PRINT_ANS_TO_FILE
    freopen("1.out", "w", stdout);
    printf("count %llu\n", sum);
    fclose(stdout);
    #endif
    //printf("count %llu\n", sum);
    //tmpTime.print("Counting time cost");
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
    stdout_fd = dup(STDOUT_FILENO);
    freopen("/dev/null", "w", stdout);
    Graph *g;
    DataLoader D;

    /*
    if (argc < 2) {
        printf("Usage: %s dataset_name graph_file [binary/text]\n", argv[0]);
        printf("Example: %s Patents ~hzx/data/patents_bin binary\n", argv[0]);
        printf("Example: %s Patents ~zms/patents_input\n", argv[0]);

        printf("\nExperimental usage: %s [graph_file.g]\n", argv[0]);
        printf("Example: %s ~hzx/data/patents.g\n", argv[0]);
        return 0;
    }

    bool binary_input = false;
    if (argc >= 4)
        binary_input = (strcmp(argv[3], "binary") == 0);

    DataType my_type;
    if (argc >= 3) {
        GetDataType(my_type, argv[1]);

        if (my_type == DataType::Invalid) {
            printf("Dataset not found!\n");
            return 0;
        }
    }*/

    using std::chrono::system_clock;
    auto t1 = system_clock::now();

    bool ok;
    /*
    if (argc >= 3) {
        // 注：load_data的第四个参数用于指定是否读取二进制文件输入，默认为false
        ok = D.load_data(g, my_type, argv[2], binary_input);
    } else {
        ok = D.fast_load(g, argv[1]);
    }
    */

    ok = D.fast_load(g, argv[1]);

    if (!ok) {
        printf("data load failure :-(\n");
        return 0;
    }

    auto t2 = system_clock::now();
    auto load_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    //printf("Load data success! time: %g seconds\n", load_time.count() / 1.0e6);
    fflush(stdout);

    // const char *pattern_str = "0111010011100011100001100"; // 5 house p1
    //const char *pattern_str = "011011101110110101011000110000101000"; // 6 p2
    // const char *pattern_str = "0111111101111111011101110100111100011100001100000"; // 7 p5
    // const char *pattern_str = "0111111101111111011001110100111100011000001100000"; // 7 p6

    int pattern_size = atoi(argv[2]);
    const char* pattern_str= argv[3];

    Pattern p(pattern_size, pattern_str);
    //printf("pattern = \n");
    //p.print();
    //printf("max intersection size %d\n", VertexSet::max_intersection_size);
    bool is_pattern_valid;
    bool use_in_exclusion_optimize = true;
    Schedule_IEP schedule_iep(p, is_pattern_valid, 0, 1, use_in_exclusion_optimize, g->v_cnt, g->e_cnt, g->tri_cnt);
    Schedule schedule(p, is_pattern_valid, 0, 1, use_in_exclusion_optimize, g->v_cnt, g->e_cnt, g->tri_cnt); // schedule is only used for getting redundancy
    schedule_iep.set_in_exclusion_optimize_redundancy(schedule.get_in_exclusion_optimize_redundancy());

    if (!is_pattern_valid) {
        printf("pattern is invalid!\n");
        return 0;
    }

    pattern_matching_init(g, schedule_iep);

    //allTime.print("Total time cost");

    return 0;
}
