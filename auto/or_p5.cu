#include <graph.h>
#include <dataloader.h>

#include <cassert>
#include <cstring>
#include <cstdint>
#include <string>
#include <algorithm>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <chrono>
using std::chrono::system_clock;

#include <gpu/config.cuh>
#include <gpu/vertex_set.cuh>

#define likely(x)   __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

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
            // __threadfence_block();
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

    // __threadfence_block();
    return ret;
}

/**
 * @brief get |A ∩ B|
 * @note when |A| > |B|, |A ∩ B| = |A| - |A - B|
 */
__device__ int get_intersection_size(const GPUVertexSet& A, const GPUVertexSet& B)
{
    int sizeA = A.get_size();
    int sizeB = B.get_size();
    if (sizeB > sizeA)
        return sizeB - unordered_subtraction_size(B, A);
    return sizeA - unordered_subtraction_size(A, B); 
}
__global__ void pattern_matching_kernel(uint32_t edge_num, uint32_t buffer_size, uint32_t *edge_from, uint32_t *edge, uint32_t *vertex, uint32_t *tmp) {
    __shared__ unsigned int block_edge_idx[WARPS_PER_BLOCK];
    extern __shared__ GPUVertexSet block_vertex_set[];
    extern __shared__ char block_shmem[];

    int wid = threadIdx.x / THREADS_PER_WARP;
    int lid = threadIdx.x % THREADS_PER_WARP;
    int global_wid = blockIdx.x * WARPS_PER_BLOCK + wid;
    unsigned int &edge_idx = block_edge_idx[wid];
    GPUVertexSet *vertex_set = block_vertex_set + wid * 5;

    GPUVertexSet &subtraction_set = vertex_set[4];
    if (lid == 0) {
        edge_idx = 0;
        uint32_t offset = buffer_size * global_wid * 4;

        uint32_t *block_subtraction_set_buf = (uint32_t *)(block_shmem + 640);
        subtraction_set.set_data_ptr(block_subtraction_set_buf + wid * 4);

        for (int i = 0; i < 4; ++i) {
            vertex_set[i].set_data_ptr(tmp + offset);
            offset += buffer_size;
        }
    }

    __threadfence_block();

    uint32_t v0, v1, v2, v3;
    uint32_t l, r;
    unsigned long long sum = 0;

    while (true) {
        if (lid == 0) {
            edge_idx = atomicAdd(&dev_cur_edge, 1);
        }
        __threadfence_block();

        unsigned int i = edge_idx;
        if (i >= edge_num) break;

        v0 = edge_from[i];
        v1 = edge[i];
        if (v0 <= v1) continue;

        get_edge_index(v0, l, r);
        if (threadIdx.x % THREADS_PER_WARP == 0)
            vertex_set[0].init(r - l, &edge[l]);
        __threadfence_block();
        
        get_edge_index(v1, l, r);
        GPUVertexSet* tmp_vset;
        intersection2(vertex_set[1].get_data_ptr(), vertex_set[0].get_data_ptr(), &edge[l], vertex_set[0].get_size(), r - l, &vertex_set[1].size);
        if (vertex_set[1].get_size() == 0) continue;
        
        int loop_size_depth2 = vertex_set[1].get_size();
        uint32_t* loop_data_ptr_depth2 = vertex_set[1].get_data_ptr();
        for (int i_depth2 = 0; i_depth2 < loop_size_depth2; ++i_depth2) {
            uint32_t v_depth2 = loop_data_ptr_depth2[i_depth2];
            if (v0 == v_depth2 || v1 == v_depth2) continue;

            unsigned int l_depth2, r_depth2;
            get_edge_index(v_depth2, l_depth2, r_depth2);
            intersection2(vertex_set[2].get_data_ptr(), vertex_set[1].get_data_ptr(), &edge[l_depth2], vertex_set[1].get_size(), r_depth2 - l_depth2, &vertex_set[2].size);
            if (vertex_set[2].get_size() == 0) continue;
            
            v2 = v_depth2; // subtraction_set.push_back(v2);

            int loop_size_depth3 = vertex_set[2].get_size();
            uint32_t* loop_data_ptr_depth3 = vertex_set[2].get_data_ptr();
            for (int i_depth3 = 0; i_depth3 < loop_size_depth3; ++i_depth3) {
                uint32_t v_depth3 = loop_data_ptr_depth3[i_depth3];
                if (v0 == v_depth3 || v1 == v_depth3 || v2 == v_depth3) continue;

                unsigned int l_depth3, r_depth3;
                get_edge_index(v_depth3, l_depth3, r_depth3);
                {
                    tmp_vset = &vertex_set[3];
                    if (threadIdx.x % THREADS_PER_WARP == 0)
                        tmp_vset->init(r_depth3 - l_depth3, &edge[l_depth3]);
                    __threadfence_block();
                    if (r_depth3 - l_depth3 > vertex_set[2].get_size())
                        tmp_vset->size -= unordered_subtraction_size(*tmp_vset, vertex_set[2], -1);
                    else
                        tmp_vset->size = vertex_set[2].get_size() - unordered_subtraction_size(vertex_set[2], *tmp_vset, -1);
                }
                if (vertex_set[3].get_size() == 0) continue;
                
                v3 = v_depth3; // subtraction_set.push_back(v3);

                int ans0 = vertex_set[3].get_size() - 0;
                int ans1 = vertex_set[2].get_size() - 1;
                int ans2 = vertex_set[1].get_size() - 2;
                long long val;
                val = ans0;
                val = val * ans1;
                val = val * ans2;
                sum += val * 1;
                val = ans0;
                val = val * ans1;
                sum += val * -1;
                val = ans0;
                val = val * ans1;
                sum += val * -1;
                val = ans0;
                val = val * ans2;
                sum += val * -1;
                val = ans0;
                sum += val * 2;
            }
        }
    }
    if (lid == 0) atomicAdd(&dev_sum, sum);
}

unsigned long long do_pattern_matching(Graph* g,
    double* p_prepare_time = nullptr, double* p_count_time = nullptr) {
    assert(g != nullptr);
    auto t1 = system_clock::now();

    cudaDeviceProp dev_props;
    cudaGetDeviceProperties(&dev_props, 0);

    int max_active_blocks_per_sm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks_per_sm,
        pattern_matching_kernel, THREADS_PER_BLOCK, 768);
    int nr_blocks = 1024;
    int nr_total_warps = nr_blocks * WARPS_PER_BLOCK;
    printf("nr_blocks=%d\n", nr_blocks);
    
    size_t size_edge = g->e_cnt * sizeof(uint32_t);
    size_t size_vertex = (g->v_cnt + 1) * sizeof(uint32_t);
    size_t size_tmp = VertexSet::max_intersection_size * sizeof(uint32_t) * nr_total_warps * 4;
    uint32_t *edge_from = new uint32_t[g->e_cnt];
    for (uint32_t i = 0; i < g->v_cnt; ++i)
        for (uint32_t j = g->vertex[i]; j < g->vertex[i+1]; ++j)
            edge_from[j] = i;

    uint32_t *dev_edge, *dev_edge_from, *dev_vertex, *dev_tmp;
    gpuErrchk( cudaMalloc((void**)&dev_edge, size_edge));
    gpuErrchk( cudaMalloc((void**)&dev_edge_from, size_edge));
    gpuErrchk( cudaMalloc((void**)&dev_vertex, size_vertex));
    gpuErrchk( cudaMalloc((void**)&dev_tmp, size_tmp));
    gpuErrchk( cudaMemcpy(dev_edge, g->edge, size_edge, cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy(dev_edge_from, edge_from, size_edge, cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy(dev_vertex, g->vertex, size_vertex, cudaMemcpyHostToDevice));

    unsigned long long sum = 0;
    unsigned cur_edge = 0;
    cudaMemcpyToSymbol(dev_sum, &sum, sizeof(sum));
    cudaMemcpyToSymbol(dev_cur_edge, &cur_edge, sizeof(cur_edge));

    auto t2 = system_clock::now();
    double prepare_time = 1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    if (p_prepare_time) *p_prepare_time = prepare_time;
    printf("prepare time: %g seconds\n", prepare_time);
    
    auto t3 = system_clock::now();
    pattern_matching_kernel<<<nr_blocks, THREADS_PER_BLOCK, 768>>>
        (g->e_cnt, VertexSet::max_intersection_size, dev_edge_from, dev_edge, dev_vertex, dev_tmp);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk( cudaMemcpyFromSymbol(&sum, dev_sum, sizeof(sum)) );

    sum /= 2; // IEP redundancy

    auto t4 = system_clock::now();
    double count_time = 1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();
    if (p_count_time) *p_count_time = count_time;
    printf("counting time: %g seconds\n", count_time);
    printf("count: %llu\n", sum);
    
    gpuErrchk(cudaFree(dev_edge));
    gpuErrchk(cudaFree(dev_edge_from));
    gpuErrchk(cudaFree(dev_vertex));
    gpuErrchk(cudaFree(dev_tmp));
    delete[] edge_from;
    return sum;
}
int main(int argc,char *argv[]) {
    Graph *g;
    DataLoader D;

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

    auto result = do_pattern_matching(g, nullptr, nullptr);
    (void) result;

    return 0;
}
