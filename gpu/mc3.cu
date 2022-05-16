#include <graph.h>
#include <dataloader.h>
#include <vertex_set.h>
#include <common.h>

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

template <typename T>
__device__ inline void swap(T& a, T& b)
{
    T t(std::move(a));
    a = std::move(b);
    b = std::move(t);
}

__device__ uint32_t get_intersection_size(const uint32_t* a, const uint32_t* b, uint32_t na, uint32_t nb)
{
    __shared__ uint32_t block_out_size[WARPS_PER_BLOCK];

    int wid = threadIdx.x / THREADS_PER_WARP; // warp id
    int lid = threadIdx.x % THREADS_PER_WARP; // lane id
    uint32_t &out_size = block_out_size[wid];

    if (na > nb) {
        swap(a, b);
        swap(na, nb);
    }

    if (lid == 0)
        out_size = 0;

    for (int num_done = 0; num_done < na; num_done += THREADS_PER_WARP) {
        if (num_done + lid < na) {
            uint32_t u = a[num_done + lid]; // u: an element in set a
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
    return out_size;
}

__device__ uint32_t dev_cur_edge = 0;
__device__ unsigned long long dev_tri_cnt = 0, dev_wedge_cnt = 0;

__global__ void motif_counting_3(uint32_t nr_edges, uint32_t* edge_from, uint32_t* edge_to, e_index_t* vertex)
{
    __shared__ e_index_t block_edge_idx[WARPS_PER_BLOCK];

    int wid = threadIdx.x / THREADS_PER_WARP;
    int lid = threadIdx.x % THREADS_PER_WARP;
    auto &edge_idx = block_edge_idx[wid];

    if (lid == 0)
        edge_idx = 0;

    uint32_t v0, v1;
    e_index_t l0, r0, l1, r1;
    uint64_t local_tri_cnt = 0, local_wedge_cnt = 0;
    while (true) {
        if (lid == 0)
            edge_idx = atomicAdd(&dev_cur_edge, 1);
        __threadfence_block();

        auto i = edge_idx;
        if (i >= nr_edges)
            break;
        
        v0 = edge_from[i];
        v1 = edge_to[i];
        l0 = vertex[v0], r0 = vertex[v0 + 1];
        if (i == l0 && lid == 0) {
            e_index_t d = r0 - l0;
            local_wedge_cnt += d * (d - 1) / 2;
        }

        if (v0 <= v1)
            continue;
        
        l1 = vertex[v1], r1 = vertex[v1 + 1];
        local_tri_cnt += get_intersection_size(&edge_to[l0], &edge_to[l1], r0 - l0, r1 - l1);
    }

    if (lid == 0) {
        atomicAdd(&dev_tri_cnt, local_tri_cnt);
        atomicAdd(&dev_wedge_cnt, local_wedge_cnt);
    }
}

__global__ void triangle_counting(uint32_t nr_edges, uint32_t* edge_from, uint32_t* edge_to, uint32_t* vertex)
{
    __shared__ uint32_t block_edge_idx[WARPS_PER_BLOCK];

    int wid = threadIdx.x / THREADS_PER_WARP;
    int lid = threadIdx.x % THREADS_PER_WARP;
    auto &edge_idx = block_edge_idx[wid];

    if (lid == 0)
        edge_idx = 0;

    uint32_t v0, v1;
    e_index_t l0, r0, l1, r1;
    uint64_t local_tri_cnt = 0;
    while (true) {
        if (lid == 0)
            edge_idx = atomicAdd(&dev_cur_edge, 1);
        __threadfence_block();

        auto i = edge_idx;
        if (i >= nr_edges)
            break;
        
        v0 = edge_from[i];
        v1 = edge_to[i];
        if (v0 <= v1)
            continue;
        
        l0 = vertex[v0], r0 = vertex[v0 + 1];
        l1 = vertex[v1], r1 = vertex[v1 + 1];
        local_tri_cnt += get_intersection_size(&edge_to[l0], &edge_to[l1], r0 - l0, r1 - l1);
    }

    if (lid == 0) {
        atomicAdd(&dev_tri_cnt, local_tri_cnt);
    }
}

double motif_counting_init(const Graph* g) {
    using namespace std::chrono;

    int device;
    gpuErrchk( cudaGetDevice(&device) );

    int nr_sms; // number of Streaming Multiprocessors
    cudaDeviceGetAttribute(&nr_sms, cudaDevAttrMultiProcessorCount, device);

    int max_active_blocks_per_sm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks_per_sm, motif_counting_3, THREADS_PER_BLOCK, 0);
    printf("max number of active warps per SM: %d\n", max_active_blocks_per_sm * WARPS_PER_BLOCK);

    int nr_blocks = max_active_blocks_per_sm * nr_sms;
    printf("nr_blocks = %d\n", nr_blocks);

    size_t size_edge = g->e_cnt * sizeof(v_index_t);
    size_t size_vertex = (g->v_cnt + 1) * sizeof(e_index_t);

    uint32_t *edge_from = new uint32_t[g->e_cnt];
    for (uint32_t i = 0; i < g->v_cnt; ++i)
        for (e_index_t j = g->vertex[i]; j < g->vertex[i+1]; ++j)
            edge_from[j] = i;

    uint32_t *dev_edge;
    uint32_t *dev_edge_from;
    e_index_t *dev_vertex;

    gpuErrchk( cudaMalloc((void**)&dev_edge, size_edge));
    gpuErrchk( cudaMalloc((void**)&dev_edge_from, size_edge));
    gpuErrchk( cudaMalloc((void**)&dev_vertex, size_vertex));

    gpuErrchk( cudaMemcpy(dev_edge, g->edge, size_edge, cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy(dev_edge_from, edge_from, size_edge, cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy(dev_vertex, g->vertex, size_vertex, cudaMemcpyHostToDevice));

    auto t1 = system_clock::now();

    motif_counting_3<<<nr_blocks, THREADS_PER_BLOCK>>>(g->e_cnt, dev_edge_from, dev_edge, dev_vertex);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    uint64_t tri_ans, wedge_ans;
    gpuErrchk( cudaMemcpyFromSymbol(&tri_ans, dev_tri_cnt, sizeof(tri_ans)) );
    gpuErrchk( cudaMemcpyFromSymbol(&wedge_ans, dev_wedge_cnt, sizeof(wedge_ans)) );

    auto t2 = system_clock::now();
    double time = duration_cast<microseconds>(t2 - t1).count() * 1e-6;

    tri_ans /= 3;

    printf("triangle: %ld wedge: %ld\n", tri_ans, wedge_ans);
    printf("counting time: %g seconds\n", time);

    gpuErrchk(cudaFree(dev_edge));
    gpuErrchk(cudaFree(dev_edge_from));
    gpuErrchk(cudaFree(dev_vertex));
    delete[] edge_from;

    return time;
}

int main(int argc,char *argv[]) {
    Graph *g;
    DataLoader D;

    using namespace std::chrono;
    auto t1 = system_clock::now();

    bool ok = D.fast_load(g, argv[1]);

    if (!ok) {
        printf("data load failure :-(\n");
        return 0;
    }

    auto t2 = system_clock::now();
    auto load_time = duration_cast<microseconds>(t2 - t1);
    printf("Load data success! time: %g seconds\n", load_time.count() / 1.0e6);
    fflush(stdout);

    motif_counting_init(g);

    return 0;
}
