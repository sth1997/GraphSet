#include <graph.h>
#include <dataloader.h>
#include <vertex_set.h>
#include <common.h>

#include <cassert>
#include <iostream>
#include <string>
#include <algorithm>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <sys/time.h>
#include <chrono>

#define THREADS_PER_BLOCK 32

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
        std::cout << title << ": " << tp_res.tv_sec << " s " << tp_res.tv_usec << " us.\n";
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

__device__ void intersection(uint32_t *tmp, uint32_t *lbases, uint32_t *rbases, uint32_t ln, uint32_t rn, uint32_t* p_tmp_size);
__device__ void intersection2(uint32_t *tmp, uint32_t *lbases, uint32_t *rbases, uint32_t ln, uint32_t rn, uint32_t* p_tmp_size);
static __device__ uint32_t do_intersection(uint32_t*, uint32_t*, uint32_t*, uint32_t, uint32_t);

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
    __device__ bool has_data (uint32_t val) const
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
    __device__ void copy(uint32_t input_size, uint32_t* input_data)//考虑改为并行
    {
        size = input_size;
        for (int i = 0; i < input_size; ++i)
            data[i] = input_data[i];
    }
    __device__ void build_vertex_set(const GPUSchedule* schedule, const GPUVertexSet* vertex_set, uint32_t* input_data, uint32_t input_size, int prefix_id)
    {
        int father_id = schedule->get_father_prefix_id(prefix_id);
        if (father_id == -1)
        {
            if(threadIdx.x == 0)
                init(input_size, input_data);
        }
        else
        {
            if(threadIdx.x == 0)
                init();
            // intersection(this->data, vertex_set[father_id].get_data_ptr(), input_data, vertex_set[father_id].get_size(), input_size, &(this->size));
            intersection2(this->data, vertex_set[father_id].get_data_ptr(), input_data, vertex_set[father_id].get_size(), input_size, &this->size);
            __syncthreads();
        }
    }

    __device__ void intersection_with(uint32_t *input_data, uint32_t input_size)
    {
        size = do_intersection(data, data, input_data, size, input_size);
    }
    /*
    __device__ void intersection_with(uint32_t *input_data, uint32_t input_size) {
        __shared__ uint32_t lblock[THREADS_PER_BLOCK], rblock[THREADS_PER_BLOCK], tmp_size;//每次都申请会不会比较浪费时间？
    
        __shared__ uint32_t cur_thread;
    
        uint32_t i = 0, j = 0;
        uint32_t lsize = THREADS_PER_BLOCK, rsize = THREADS_PER_BLOCK;
    
        bool hit;
    
        if( threadIdx.x == 0 ) {
            tmp_size = 0;
        }
        __syncthreads();
    
        while (i < input_size && j < size) {
    
            lsize = min(input_size - i, THREADS_PER_BLOCK);
            rsize = min(size - j, THREADS_PER_BLOCK);
    
            if(i + threadIdx.x < input_size) lblock[threadIdx.x] = input_data[i + threadIdx.x];
            if(j + threadIdx.x < size) rblock[threadIdx.x] = data[j + threadIdx.x];
    
            __threadfence_block();
    
            hit = false;
            //之后考虑根据集合大小来选择循环lsize还是rsize
            for(int k = 0; k < rsize; ++k)
                hit |= (threadIdx.x < lsize) & (lblock[threadIdx.x] == rblock[k]);
            
            if( threadIdx.x == 0) {
                cur_thread = 0;
            }
            __syncthreads();
    
            while( cur_thread < THREADS_PER_BLOCK) { //这里完全是线性的，之后能不能优化？
                if(cur_thread == threadIdx.x) {
                    if(hit && i + threadIdx.x < input_size)
                        data[tmp_size++] = lblock[threadIdx.x];
                    ++cur_thread;
                }
                __syncthreads();
            }
            
            uint32_t llast = lblock[lsize - 1];
            uint32_t rlast = rblock[rsize - 1];
    
            if(llast >= rlast) j += rsize;
            if(llast <= rlast) i += lsize;
        }

        if( threadIdx.x == 0 ) {
            size = tmp_size;
        }
    }
    */

private:
    uint32_t size;
    uint32_t* data;
};

__device__ unsigned long long dev_sum;
__device__ unsigned int dev_nowEdge;

/**
 * merge-based set intersection
 */
__device__ void intersection(uint32_t *tmp, uint32_t *lbases, uint32_t *rbases, uint32_t ln, uint32_t rn, uint32_t* p_tmp_size) {
    __shared__ uint32_t lblock[THREADS_PER_BLOCK];//每次都申请会不会比较浪费时间？
    __shared__ uint32_t rblock[THREADS_PER_BLOCK];

    __shared__ uint32_t cur_thread;

    uint32_t i = 0, j = 0;
    uint32_t lsize = THREADS_PER_BLOCK, rsize = THREADS_PER_BLOCK;

    bool hit;

    if( threadIdx.x == 0 )
        *p_tmp_size = 0;
    __syncthreads();

    while (i < ln && j < rn) {

        lsize = min(ln - i, THREADS_PER_BLOCK);
        rsize = min(rn - j, THREADS_PER_BLOCK);

        if(i + threadIdx.x < ln) lblock[threadIdx.x] = lbases[i + threadIdx.x];
        if(j + threadIdx.x < rn) rblock[threadIdx.x] = rbases[j + threadIdx.x];

        __threadfence_block();

        hit = false;
        for(int k = 0; k < rsize; ++k)
            hit |= (threadIdx.x < lsize) & (lblock[threadIdx.x] == rblock[k]);
        
        if( threadIdx.x == 0) {
            cur_thread = 0;
        }
        __syncthreads();

        while( cur_thread < THREADS_PER_BLOCK) { //这里完全是线性的，之后能不能优化？
            if(cur_thread == threadIdx.x) {
                if(hit && i + threadIdx.x < ln)
                    tmp[(*p_tmp_size)++] = lblock[threadIdx.x];
                ++cur_thread;
            }
            __syncthreads();
        }
        
        uint32_t llast = lblock[lsize - 1];
        uint32_t rlast = rblock[rsize - 1];

        if(llast >= rlast) j += rsize;
        if(llast <= rlast) i += lsize;//考虑改为在这两个分支中修改block
    }

/*    i = 0;
    j = 0;
    int size = 0;
    while(i < ln && j < rn) {
        if(lbases[i]==rbases[j]) {
            assert(lbases[i]==tmp[size]);
            ++size;
        }
        int u = lbases[i],v=rbases[j];
        i+=u<=v;
        j+=v<=u;
    }
    assert(size==*p_tmp_size);*/
}

/**
 * search-based intersection
 * 
 * returns the size of the intersection set
 */
__device__ uint32_t do_intersection(uint32_t* out, uint32_t* a, uint32_t* b, uint32_t na, uint32_t nb)
{
    __shared__ uint32_t out_offset[THREADS_PER_BLOCK];
    __shared__ uint32_t out_size;

    if (threadIdx.x == 0) {
        out_size = 0;
    }

    uint32_t blk_size, i = 0;
    while (i < na) {
        blk_size = min(na - i, THREADS_PER_BLOCK);

        bool found = 0;
        uint32_t u = 0;
        if (threadIdx.x < blk_size) {
            int mid, l = 0, r = nb - 1; // [l, r], use signed int instead of unsigned int!
            u = a[i + threadIdx.x]; // u: an element in set a
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
        out_offset[threadIdx.x] = found;
        __syncthreads();

        // currently blockDim.x == THREADS_PER_BLOCK
        for (int s = 1; s < blockDim.x; s *= 2) {
            int index = threadIdx.x;
            if (index >= s) {
                out_offset[index] += out_offset[index - s];
            }
            __syncthreads();
        }

        if (found) {
            uint32_t offset = out_offset[threadIdx.x] - 1;
            out[out_size + offset] = u;
        }

        if (threadIdx.x == 0) {
            out_size += out_offset[THREADS_PER_BLOCK - 1];
        }
        __syncthreads();

        i += blk_size;
    }

    return out_size;
}

/**
 * wrapper of search based intersection `do_intersection`
 */
__device__ void intersection2(uint32_t *tmp, uint32_t *lbases, uint32_t *rbases, uint32_t ln, uint32_t rn, uint32_t* p_tmp_size)
{
    // make sure ln <= rn
    if (ln > rn) {
        swap(ln, rn);
        swap(lbases, rbases);
    }

    uint32_t intersection_size = do_intersection(tmp, lbases, rbases, ln, rn);

    if (threadIdx.x == 0)
        *p_tmp_size = intersection_size;
}

//set0 - set1
__device__ int unorderd_subtraction_size(const GPUVertexSet& set0, const GPUVertexSet& set1, int size_after_restrict = -1)
{
    int size0 = set0.get_size();
    int size1 = set1.get_size();
    if (size_after_restrict != -1)
        size0 = size_after_restrict;
    

    __shared__ int ret;
    if (threadIdx.x == 0)
        ret = size0;
    __syncthreads();

    int done1 = 0;
    while (done1 < size1)
    {
        if (threadIdx.x < size1)
        {
            int l = 0, r = size0 - 1;
            int val = set1.get_data(threadIdx.x + done1);
            //考虑之后换一下二分查找的写法，比如改为l < r，然后把mid的判断从循环里去掉，放到循环外(即最后l==r的时候)
            while (l <= r)
            {
                int mid = (l + r) >> 1;
                if (set0.get_data(mid) == val)
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
        done1 += THREADS_PER_BLOCK;
    }
    __syncthreads();
    return ret;
}

__device__ void GPU_pattern_matching_aggressive_func(const GPUSchedule* schedule, GPUVertexSet* vertex_set, GPUVertexSet& subtraction_set, GPUVertexSet& tmp_set, unsigned long long& local_ans, int depth, uint32_t *edge, uint32_t *vertex)
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
        int loop_set_prefix_ids[8];//偷懒用了static，之后考虑改成dynamic
        loop_set_prefix_ids[0] = loop_set_prefix_id;
        for(int i = 1; i < in_exclusion_optimize_num; ++i)
            loop_set_prefix_ids[i] = schedule->get_loop_set_prefix_id( depth + i );
        /*if (threadIdx.x == 0 && subtraction_set.get_data(0) == 2 && subtraction_set.get_data(1) == 1 && subtraction_set.get_data(2) == 0)
        {
            for(int i = 0; i < in_exclusion_optimize_num; ++i)
            {
                printf("id = %d   set size = %d   data = ", schedule->get_loop_set_prefix_id( depth + i ), vertex_set[schedule->get_loop_set_prefix_id( depth + i )].get_size());
                for (int j = 0; j < vertex_set[schedule->get_loop_set_prefix_id( depth + i )].get_size(); ++j)
                    printf("%d ", vertex_set[schedule->get_loop_set_prefix_id( depth + i )].get_data(j));
                printf("\n");
            }
            printf("in ex group size = %d\n", schedule->in_exclusion_optimize_group.size);
        }*/
        for(int optimize_rank = 0; optimize_rank < schedule->in_exclusion_optimize_group.size; ++optimize_rank) {
            const GPUGroupDim1& cur_graph = schedule->in_exclusion_optimize_group.data[optimize_rank];
            long long val = schedule->in_exclusion_optimize_val[optimize_rank];
            //if (threadIdx.x == 0 && subtraction_set.get_data(0) == 2 && subtraction_set.get_data(1) == 1 && subtraction_set.get_data(2) == 0)
            //    printf("origin val = %lld\n", val);
            for(int cur_graph_rank = 0; cur_graph_rank < cur_graph.size; ++cur_graph_rank) {
                if(cur_graph.data[cur_graph_rank].size == 1) {
                    int id = loop_set_prefix_ids[cur_graph.data[cur_graph_rank].data[0]];
                    //val = val * unorderd_subtraction_size(vertex_set[id], subtraction_set);
                    int tmp = unorderd_subtraction_size(vertex_set[id], subtraction_set);
                    //if (threadIdx.x == 0 && subtraction_set.get_data(0) == 2 && subtraction_set.get_data(1) == 1 && subtraction_set.get_data(2) == 0)
                    //    printf("data size = 1   id = %d  tmp = %d\n", id, tmp);
                    val = val * tmp;
                }
                else {
                    int id = loop_set_prefix_ids[cur_graph.data[cur_graph_rank].data[0]];
                    tmp_set.copy(vertex_set[id].get_size(), vertex_set[id].get_data_ptr());

                    for(int i = 1; i < cur_graph.data[cur_graph_rank].size; ++i) {
                        int id = loop_set_prefix_ids[cur_graph.data[cur_graph_rank].data[i]];
                        tmp_set.intersection_with(vertex_set[id].get_data_ptr(), vertex_set[id].get_size());
                    }
                    /*if (threadIdx.x == 0 && subtraction_set.get_data(0) == 2 && subtraction_set.get_data(1) == 1 && subtraction_set.get_data(2) == 4)
                    {
                        printf("data size = %d   id = ", cur_graph.data[cur_graph_rank].size);
                        for (int i = 0; i < cur_graph.data[cur_graph_rank].size; ++i)
                            printf("%d ", loop_set_prefix_ids[cur_graph.data[cur_graph_rank].data[i]]);
                        printf("\n");
                        printf("tmp_set.size = %d  data = %d\n", tmp_set.get_size(), tmp_set.get_data(0));
                    }*/
                    val = val * unorderd_subtraction_size(tmp_set, subtraction_set);
                }
                if (val == 0)
                    break;
            }
            //if (threadIdx.x == 0 && subtraction_set.get_data(0) == 2 && subtraction_set.get_data(1) == 1 && subtraction_set.get_data(2) == 4)
            //    printf("val = %lld\n", val);
            
            /*if (threadIdx.x == 0 && val != 0) {
                printf("val=%lld    v0 = %d v1 = %d v2 = %d\n", val, subtraction_set.get_data(0), subtraction_set.get_data(1), subtraction_set.get_data(2));
                for(int i = 0; i < in_exclusion_optimize_num; ++i)
                {
                    printf("id = %d   set size = %d   data = ", schedule->get_loop_set_prefix_id( depth + i ), vertex_set[schedule->get_loop_set_prefix_id( depth + i )].get_size());
                    for (int j = 0; j < vertex_set[schedule->get_loop_set_prefix_id( depth + i )].get_size(); ++j)
                        printf("%d ", vertex_set[schedule->get_loop_set_prefix_id( depth + i )].get_data(j));
                    printf("\n");
                }
                for(int cur_graph_rank = 0; cur_graph_rank < cur_graph.size; ++cur_graph_rank) {
                    if(cur_graph.data[cur_graph_rank].size == 1) {
                        printf("data size = 1   id = %d\n", loop_set_prefix_ids[cur_graph.data[cur_graph_rank].data[0]]);
                    }
                    else {
                        printf("data size = %d   id = ", cur_graph.data[cur_graph_rank].size);
                        for (int i = 0; i < cur_graph.data[cur_graph_rank].size; ++i)
                            printf("%d ", loop_set_prefix_ids[cur_graph.data[cur_graph_rank].data[i]]);
                        printf("\n");
                    }
                }
            }*/
            if (threadIdx.x == 0)
                local_ans += val;
        }
        return;
    }

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
        if(threadIdx.x == 0)
            subtraction_set.push_back(v);
        __syncthreads();
        GPU_pattern_matching_aggressive_func(schedule, vertex_set, subtraction_set, tmp_set, local_ans, depth + 1, edge, vertex);
        if(threadIdx.x == 0)
            subtraction_set.pop_back();
        __syncthreads();
    }
}

__global__ void gpu_pattern_matching(uint32_t edge_num, uint32_t buffer_size, uint32_t *edge_from, uint32_t *edge, uint32_t *vertex, uint32_t *tmp, const GPUSchedule* schedule) {
    __shared__ unsigned int edgeI;
    __shared__ unsigned int edgeEnd;
    __shared__ unsigned long long sdata[THREADS_PER_BLOCK]; 
    //之后考虑把tmp buffer都放到shared里来（如果放得下）
    extern __shared__ GPUVertexSet vertex_set[];

    unsigned long long local_sum = 0;

    //uint32_t tmp1_begin = buffer_size * 2 * blockIdx.x;
    //uint32_t tmp2_begin = tmp1_begin + buffer_size;
    
    if(threadIdx.x == 0) {
        edgeI = edgeEnd = 0;
        uint32_t offset = buffer_size * blockIdx.x * (schedule->get_total_prefix_num() + 1);
        for (int i = 0; i < schedule->get_total_prefix_num() + 2; ++i)
        {
            vertex_set[i].set_data_ptr(tmp + offset);
            offset += buffer_size;
        }
    }
    GPUVertexSet& subtraction_set = vertex_set[schedule->get_total_prefix_num()];
    GPUVertexSet& tmp_set = vertex_set[schedule->get_total_prefix_num() + 1];

    __syncthreads(); //之后考虑把所有的syncthreads都改成syncwarp
    
    assert( edgeI == edgeEnd);

    //uint32_t v0,v1,v2;
    //__shared__ uint32_t tmp1_size;
    //__shared__ uint32_t tmp2_size;
    uint32_t v0, v1;

    //uint32_t lb,le,ln;
    //uint32_t rb,re,rn;
    uint32_t l, r;

    //uint32_t loop_begin, loop_limit;

    //__shared__ bool have_v2;

    while(true) {
        if(threadIdx.x == 0) {
            //printf("%d at %u\n", blockIdx.x, edgeI);
            if(++edgeI >= edgeEnd) { //这个if语句应该是每次都会发生吧？
                edgeI = atomicAdd(&dev_nowEdge, 1);
                edgeEnd = min(edge_num, edgeI + 1); //这里不需要原子读吗
                unsigned int i = edgeI;
                if (i < edge_num)
                {
                    subtraction_set.init();
                    subtraction_set.push_back(edge_from[i]);
                    subtraction_set.push_back(edge[i]);
                }
            }
        }

        __syncthreads();

        unsigned int i = edgeI;
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
        //if (v0 == 2 && v1 == 1 && threadIdx.x == 0)
        //    printf("v0 = 2 and v1 = 1\n");
        
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
        
        GPU_pattern_matching_aggressive_func(schedule, vertex_set, subtraction_set, tmp_set, local_sum, 2, edge, vertex);

        /*

        intersection(tmp + tmp1_begin, edge + lb, edge + rb, ln, rn, &tmp1_size); // v3's set = tmp1 = N(v0) & N(v1)
        __syncthreads();
        
        if(tmp1_size == 0) continue;

        loop_begin = vertex[v1];
        loop_limit = vertex[v1+1];
        for(uint32_t j = loop_begin; j < loop_limit; ++j) {
            v2 = edge[j]; // for v2 in N(v1)
            if(v0==v2) continue;

            rb = vertex[v2];
            re = vertex[v2+1];
            rn = re - rb;

            intersection(tmp + tmp2_begin, edge + lb, edge + rb, ln, rn, &tmp2_size); // v4's set = tmp2 = N(v0) & N(v2)
            __syncthreads();

            if(tmp2_size <= 1) continue;

            if(threadIdx.x == 0) {
                have_v2 = false;
            }
            __syncthreads();

            detect_v2(tmp + tmp1_begin, tmp1_size, v2, &have_v2); // notice that v2 may belong to tmp1, but we want tmp1 - {v2}

            upd_ans(tmp + tmp1_begin, tmp + tmp2_begin, tmp1_size, tmp2_size, &local_sum); // ans -= (tmp1 & tmp2).size
            __syncthreads();

            if(threadIdx.x == 0) { // ans += tmp1.size * tmp2.size, notice that v1 always exist in tmp2, so we use tmp2_size-1
                if(have_v2) local_sum += (tmp1_size - 1) * (tmp2_size - 1);
                else local_sum += tmp1_size * (tmp2_size - 1);
            }
        }*/
    }

    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    for (int s=1; s < blockDim.x; s *=2){
        int index = 2 * s * threadIdx.x;

        if (index < blockDim.x){
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(&dev_sum, sdata[0]);
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

    //printf("in ex num = %d\n", schedule.get_in_exclusion_optimize_num());
    //printf("total restrict num = %d\n", schedule.get_total_restrict_num());
    //printf("restrict_last[1] = %d   index = %d\n", schedule.get_restrict_last(1), schedule.get_restrict_index(schedule.get_restrict_last(1)));

    tmpTime.check(); 
    int numBlocks = 4096;

    uint32_t size_edge = g->e_cnt * sizeof(uint32_t);
    uint32_t size_vertex = (g->v_cnt + 1) * sizeof(uint32_t);
    uint32_t size_tmp  = VertexSet::max_intersection_size * sizeof(uint32_t) * numBlocks * (schedule.get_total_prefix_num() + 2); //prefix + subtraction + tmp

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

    gpuErrchk( cudaMemcpyToSymbol(dev_sum, &sum, sizeof(unsigned long long), 0, cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpyToSymbol(dev_nowEdge, &sum, sizeof(unsigned int), 0, cudaMemcpyHostToDevice));

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
    

    /*gpuErrchk( cudaMemcpy(dev_schedule, &schedule, sizeof(Schedule), cudaMemcpyHostToDevice));
    
    int *dev_schedule_adj_mat, *dev_schedule_father_prefix_id;
    int *dev_schedule_last, *dev_schedule_next, *dev_schedule_loop_set_prefix_id;
    int *dev_schedule_restrict_last, *dev_schedule_restrict_next, *dev_schedule_restrict_index;
    //考虑之后只malloc一个大buffer，然后分别去分配，这样访存局部性也许会好一些？
    gpuErrchk( cudaMalloc((void**)&dev_schedule_adj_mat, sizeof(int) * schedule_size * schedule_size));
    gpuErrchk( cudaMemcpy(dev_schedule_adj_mat, schedule.get_adj_mat_ptr(), sizeof(int) * schedule_size * schedule_size, cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy(&dev_schedule->adj_mat, &dev_schedule_adj_mat, sizeof(int*), cudaMemcpyHostToDevice));

    gpuErrchk( cudaMalloc((void**)&dev_schedule_father_prefix_id, sizeof(int) * max_prefix_num));
    gpuErrchk( cudaMemcpy(dev_schedule_father_prefix_id, schedule.get_father_prefix_id_ptr(), sizeof(int) * max_prefix_num, cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy(&dev_schedule->father_prefix_id, &dev_schedule_father_prefix_id, sizeof(int*), cudaMemcpyHostToDevice));

    gpuErrchk( cudaMalloc((void**)&dev_schedule_last, sizeof(int) * schedule_size));
    gpuErrchk( cudaMemcpy(dev_schedule_last, schedule.get_last_ptr(), sizeof(int) * schedule_size, cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy(&dev_schedule->last, &dev_schedule_last, sizeof(int*), cudaMemcpyHostToDevice));

    gpuErrchk( cudaMalloc((void**)&dev_schedule_next, sizeof(int) * max_prefix_num));
    gpuErrchk( cudaMemcpy(dev_schedule_next, schedule.get_next_ptr(), sizeof(int) * max_prefix_num, cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy(&dev_schedule->next, &dev_schedule_next, sizeof(int*), cudaMemcpyHostToDevice));

    gpuErrchk( cudaMalloc((void**)&dev_schedule_loop_set_prefix_id, sizeof(int) * schedule_size));
    gpuErrchk( cudaMemcpy(dev_schedule_loop_set_prefix_id, schedule.get_loop_set_prefix_id_ptr(), sizeof(int) * schedule_size, cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy(&dev_schedule->loop_set_prefix_id, &dev_schedule_loop_set_prefix_id, sizeof(int*), cudaMemcpyHostToDevice));

    gpuErrchk( cudaMalloc((void**)&dev_schedule_restrict_last, sizeof(int) * schedule_size));
    gpuErrchk( cudaMemcpy(dev_schedule_restrict_last, schedule.get_restrict_last_ptr(), sizeof(int) * schedule_size, cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy(&dev_schedule->restrict_last, &dev_schedule_restrict_last, sizeof(int*), cudaMemcpyHostToDevice));

    gpuErrchk( cudaMalloc((void**)&dev_schedule_restrict_next, sizeof(int) * max_prefix_num));
    gpuErrchk( cudaMemcpy(dev_schedule_restrict_next, schedule.get_restrict_next_ptr(), sizeof(int) * max_prefix_num, cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy(&dev_schedule->restrict_next, &dev_schedule_restrict_next, sizeof(int*), cudaMemcpyHostToDevice));

    gpuErrchk( cudaMalloc((void**)&dev_schedule_restrict_index, sizeof(int) * max_prefix_num));
    gpuErrchk( cudaMemcpy(dev_schedule_restrict_index, schedule.get_restrict_index_ptr(), sizeof(int) * max_prefix_num, cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy(&dev_schedule->restrict_index, &dev_schedule_restrict_index, sizeof(int*), cudaMemcpyHostToDevice));*/

    tmpTime.print("Prepare time cost");
    tmpTime.check();

    uint32_t edge_num = g->e_cnt;
    uint32_t buffer_size = VertexSet::max_intersection_size;
    //因为目前用了managed开内存，所以第一次运行kernel会有一定额外开销，考虑运行两次，第一次作为warmup
    gpu_pattern_matching<<<numBlocks, THREADS_PER_BLOCK, (schedule.get_total_prefix_num() + 2) * sizeof(GPUVertexSet)>>>(edge_num, buffer_size, dev_edge_from, dev_edge, dev_vertex, dev_tmp, dev_schedule);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk( cudaMemcpyFromSymbol(&sum, dev_sum, sizeof(sum)) );

    printf("house count %llu\n", sum);
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

    assert(D.load_data(g,my_type,path.c_str())==true); 

 //   assert(D.load_data(g,10));
    printf("Load data success!\n");
    fflush(stdout);

    allTime.check();

    // using std::chrono::system_clock;
    // auto t1 = system_clock::now();

    //Pattern p(PatternType::Cycle_6_Tri);
    char tmpbuf[100] = "011110101101110000110000100001010010";
    Pattern p(6, tmpbuf);
    printf("pattern = \n");
    p.print();
    printf("max intersection size %d\n", VertexSet::max_intersection_size);
    bool is_pattern_valid;
    bool use_in_exclusion_optimize = true;
    Schedule schedule(p, is_pattern_valid, 1, 1, use_in_exclusion_optimize, g->v_cnt, g->e_cnt, g->tri_cnt);
    //Schedule schedule(p, is_pattern_valid, 0, 1, use_in_exclusion_optimize, g->v_cnt, g->e_cnt, g->tri_cnt); // use the best schedule
    assert(is_pattern_valid);

    pattern_matching_init(g, schedule);

    // auto t2 = system_clock::now();
    // auto elapsed = t2 - t1;
    // auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed);
    // std::cout << "total time (chrono): " << ms.count() << "ms\n";
    allTime.print("Total time cost");

    return 0;
}

