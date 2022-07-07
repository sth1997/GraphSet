/**
 * 这个版本里面没有细粒度计时。有计时的在gpu_graph_with_timer.cu里面。
 * 而且计时的方式与zms版本略有区别。
 */
#include <graph.h>
#include <dataloader.h>
#include <vertex_set.h>
#include <common.h>
#include <schedule_IEP.h>
#include <motif_generator.h>

#include <cassert>
#include <cstring>
#include <cstdint>
#include <string>
#include <algorithm>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "timeval.h"
#include "intersection.cuh"

#define TYPE3 1


#define likely(x)   __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
constexpr int THREADS_PER_BLOCK = 32;
constexpr int THREADS_PER_WARP = 32;
constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / THREADS_PER_WARP;

//#define PRINT_ANS_TO_FILE //用于scripts/small_graph_check.py


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


__device__ void intersection2(uint32_t *out, uint32_t *tmp, const uint32_t *lbases, const uint32_t *rbases, uint32_t ln, uint32_t rn, uint32_t* p_tmp_size);


__device__ uint32_t binary_search_intersection(uint32_t* out, uint32_t* tmp, const uint32_t* a, const uint32_t* b, uint32_t na, uint32_t nb)
{
    __shared__ uint32_t block_out_offset[THREADS_PER_BLOCK];
    __shared__ uint32_t block_out_size[WARPS_PER_BLOCK];

    int wid = threadIdx.x / THREADS_PER_WARP; // warp id
    uint32_t *out_offset = block_out_offset + wid * THREADS_PER_WARP;
    uint32_t &out_size = block_out_size[wid];
    return warp_binary_search_intersection(out_offset, out_size, out, a, b, na, nb);
}

__device__ uint32_t serial_merge_intersection(uint32_t* out, uint32_t* tmp, const uint32_t* a, const uint32_t* b, uint32_t na, uint32_t nb)
{
    uint32_t out_size = 0;
    if (threadIdx.x % warpSize == 0) {
        out_size = merge(out, a, b, na, nb);
        // printf("out_size: %d\n", out_size);
    }
    return out_size;
}

__device__ uint32_t parallel_merge_intersection(uint32_t* out, uint32_t* tmp, const uint32_t* a, const uint32_t* b, uint32_t na, uint32_t nb)
{
    __shared__ uint32_t block_out_offset[THREADS_PER_BLOCK];
    __shared__ uint32_t block_border[THREADS_PER_BLOCK];

    int wid = threadIdx.x / warpSize;
    auto out_offset = block_out_offset + wid * warpSize;
    auto border = block_border + wid * warpSize;
    return warp_parallel_merge_intersection(out_offset, border, out, tmp, a, b, na, nb);
}


__device__ void intersection2(uint32_t *out, uint32_t *tmp, const uint32_t *lbases, const uint32_t *rbases, uint32_t ln, uint32_t rn, uint32_t &ans)
{
    // make sure ln <= rn
    if (ln > rn) {
        swap(ln, rn);
        swap(lbases, rbases);
    }
    uint32_t intersection_size;
    #ifdef TYPE1
        intersection_size = binary_search_intersection(out, tmp, lbases, rbases, ln, rn);
    #endif
    #ifdef TYPE2
        intersection_size = serial_merge_intersection(out, tmp, lbases, rbases, ln, rn);
    #endif
    #ifdef TYPE3
        intersection_size = parallel_merge_intersection(out, tmp, lbases, rbases, ln, rn);
    #endif
    __threadfence_block();
    
    // if (threadIdx.x % THREADS_PER_WARP == 0) {
    //     // printf("intersection_size: %d\n", intersection_size);
    //     ans = ans + intersection_size;
    // }

    // __threadfence_block();
}


__global__ void test_intersection(int t, int m_cnt, uint32_t* m_vertex, int n_cnt, uint32_t* n_vertex, uint32_t vertex_num, e_index_t edge_num, uint32_t* edge_from, uint32_t *edge, e_index_t *vertex, uint32_t* tmp) {
    uint32_t* out = tmp + vertex_num;
    __shared__ uint32_t ans;

    if(threadIdx.x == 0){
        ans = 0;
    }

    for(int x = 0; x < t; x++){
        for(int i = 0; i < m_cnt; i++){
            for(int j = 0; j < n_cnt; j++){
                int m_v = m_vertex[i];
                int n_v = n_vertex[j];
                intersection2(out,tmp, &edge[vertex[m_v]], &edge[vertex[n_v]], vertex[m_v + 1] - vertex[m_v], vertex[n_v+1] - vertex[n_v], ans);
            }
        }
    }
    // if(threadIdx.x == 0)
    //     printf("sum: %d\n",ans);
}

void test_intersection(Graph *g, int t, int m, int n) {
    cudaFree(NULL);

    int num_blocks = 1;

    size_t size_edge = g->e_cnt * sizeof(uint32_t);
    size_t size_vertex = (g->v_cnt + 1) * sizeof(e_index_t);

    uint32_t *edge_from = new uint32_t[g->e_cnt];
    for(uint32_t i = 0; i < g->v_cnt; ++i)
        for(uint32_t j = g->vertex[i]; j < g->vertex[i+1]; ++j)
            edge_from[j] = i;



    uint32_t *m_vertex = new uint32_t[g->v_cnt], * n_vertex = new uint32_t[g->v_cnt];
    int m_size = 0, n_size = 0;

    for(int i = 0; i < g->v_cnt; i++) {
        if(g->vertex[i+1] - g->vertex[i] == m){
            m_vertex[m_size++] = i;
        }
        if(g->vertex[i+1] - g->vertex[i] == n){
            n_vertex[n_size++] = i;
        }
    }

    for(int i = 1; i <= 1000000; i++){
        int t1 = rand() % m_size, t2 = rand() % m_size;
        uint32_t tmp = m_vertex[t1];
        m_vertex[t1] = m_vertex[t2];
        m_vertex[t2] = tmp; 
    }

    for(int i = 1; i <= 1000000; i++){
        int t1 = rand() % n_size, t2 = rand() % n_size;
        uint32_t tmp = n_vertex[t1];
        n_vertex[t1] = n_vertex[t2];
        n_vertex[t2] = tmp; 
    }

    m_size = min(m_size, 1000);
    n_size = min(n_size, 1000);


    printf("m:%d cnt: %d , n:%d vertex count:%d\n", m, m_size, n, n_size);

    uint32_t *dev_edge;
    uint32_t *dev_edge_from;
    e_index_t *dev_vertex;
    uint32_t *dev_tmp;
    uint32_t *dev_m_vertex;
    uint32_t *dev_n_vertex;

    gpuErrchk( cudaMalloc((void**)&dev_edge, size_edge));
    gpuErrchk( cudaMalloc((void**)&dev_edge_from, size_edge));
    gpuErrchk( cudaMalloc((void**)&dev_vertex, size_vertex));
    gpuErrchk( cudaMalloc((void**)&dev_m_vertex, g->v_cnt * sizeof(uint32_t)));
    gpuErrchk( cudaMalloc((void**)&dev_n_vertex, g->v_cnt * sizeof(uint32_t)));
    gpuErrchk( cudaMalloc((void**)&dev_tmp, (g->v_cnt*2) * sizeof(uint32_t)));


    

    gpuErrchk( cudaMemcpy(dev_edge, g->edge, size_edge, cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy(dev_edge_from, edge_from, size_edge, cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy(dev_vertex, g->vertex, size_vertex, cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy(dev_m_vertex, m_vertex, g->v_cnt * sizeof(uint32_t), cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy(dev_n_vertex, n_vertex, g->v_cnt * sizeof(uint32_t), cudaMemcpyHostToDevice));

    tmpTime.check(); 
    
    test_intersection<<<num_blocks, THREADS_PER_BLOCK>>>
        (t, m_size, dev_m_vertex, n_size, dev_n_vertex, g->v_cnt, g->e_cnt, dev_edge_from, dev_edge, dev_vertex, dev_tmp);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    
    #ifdef PRINT_ANS_TO_FILE
    freopen("1.out", "w", stdout);
    printf("count %llu\n", sum);
    fclose(stdout);
    #endif

    tmpTime.print("Counting time cost");
    //之后需要加上cudaFree

    // 尝试释放一些内存
    gpuErrchk(cudaFree(dev_edge));
    gpuErrchk(cudaFree(dev_edge_from));
    gpuErrchk(cudaFree(dev_vertex));
    gpuErrchk(cudaFree(dev_tmp));
    gpuErrchk(cudaFree(dev_m_vertex));
    gpuErrchk(cudaFree(dev_n_vertex));



    delete[] edge_from;
    delete[] m_vertex;
    delete[] n_vertex;
}

int main(int argc,char *argv[]) {
    srand(time(0));
    #ifdef TYPE1
        printf("type1.\n");
    #endif
    #ifdef TYPE2
        printf("type2.\n");
    #endif
    #ifdef TYPE3
        printf("type3.\n");
    #endif
    Graph *g;
    DataLoader D;

    
    if (argc < 2) {
        printf("Usage: %s dataset_name graph_file [binary/text]\n", argv[0]);
        printf("Example: %s Patents ~hzx/data/patents_bin binary\n", argv[0]);
        printf("Example: %s Patents ~zms/patents_input\n", argv[0]);

        printf("\nExperimental usage: %s [graph_file.g]\n", argv[0]);
        printf("Example: %s ~hzx/data/patents.g\n", argv[0]);
        return 0;
    }

    // bool binary_input = false;
    // if (argc >= 4)
    //     binary_input = (strcmp(argv[3], "binary") == 0);

    // DataType my_type;
    // if (argc >= 3) {
    //     GetDataType(my_type, argv[1]);

    //     if (my_type == DataType::Invalid) {
    //         printf("Dataset not found!\n");
    //         return 0;
    //     }
    // }

    using std::chrono::system_clock;
    auto t1 = system_clock::now();

    bool ok;

    // if (argc >= 3) {
    //     // 注：load_data的第四个参数用于指定是否读取二进制文件输入，默认为false
    //     ok = D.load_data(g, my_type, argv[2], binary_input);
    // } else {
    //     ok = D.fast_load(g, argv[1]);
    // }


        
    ok = D.fast_load(g, argv[1]);

    if (!ok) {
        printf("data load failure :-(\n");
        return 0;
    }

    int t,m,n;

    t = atoi(argv[2]);
    m = atoi(argv[3]);
    n = atoi(argv[4]);


    auto t2 = system_clock::now();
    auto load_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    // printf("Load data success! time: %g seconds\n", load_time.count() / 1.0e6);
    fflush(stdout);

    allTime.check();

    /*
    MotifGenerator mg(3);
    std::vector<Pattern> motifs = mg.generate();
    printf("motifs number = %d\n", motifs.size());
    for (int i = 0; i < motifs.size(); ++i) {
    //for (int i = motifs.size() - 1; i >= 0; --i) {
    Pattern p = motifs[i];
    */
    // printf("max intersection size %d\n", VertexSet::max_intersection_size);


    test_intersection(g, t, m, n);

    // allTime.print("Total time cost");
    return 0;
}
