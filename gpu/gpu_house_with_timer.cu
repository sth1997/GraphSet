// This program can only count House pattern using GPU.
#include <graph.h>
#include <dataloader.h>
#include <vertex_set.h>
#include <common.h>

#include <assert.h>
#include <iostream>
#include <string>
#include <algorithm>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <sys/time.h>
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

#define COUNT_INTERSECTION 0
#define COUNT_TIME 1

#if COUNT_TIME
__device__ uint32_t i_count = 0;
__device__ uint64_t i_time = 0;

__device__ static inline uint64_t read_timer()
{
    // Due to a bug in CUDA's 64-bit globaltimer, the lower 32 bits can wrap
    // around after the upper bits have already been read. Work around this by
    // reading the high bits a second time. Use the second value to detect a
    // rollover, and set the lower bits of the 64-bit "timer reading" to 0, which
    // would be valid, it's passed over during the duration of the reading. If no
    // rollover occurred, just return the initial reading.
    volatile uint64_t first_reading;
    volatile uint32_t second_reading;
    uint32_t high_bits_first;
    asm volatile ("mov.u64 %0, %%globaltimer;" : "=l"(first_reading));
    high_bits_first = first_reading >> 32;
    asm volatile ("mov.u32 %0, %%globaltimer_hi;" : "=r"(second_reading));
    if (high_bits_first == second_reading) {
        return first_reading;
    }
    // Return the value with the updated high bits, but the low bits set to 0.
    return ((uint64_t) second_reading) << 32;
}
#endif

#if COUNT_INTERSECTION
__device__ uint64_t i_inter_size_0 = 0, i_inter_time_0 = 0;
__device__ uint64_t i_inter_size_1 = 0, i_inter_time_1 = 0;
__device__ uint64_t i_inter_size_2 = 0, i_inter_time_2 = 0;
#endif

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define THREADS_PER_BLOCK 32

__device__ unsigned long long dev_sum;
__device__ unsigned int dev_nowEdge;

__device__ void intersection(uint32_t *out, uint32_t *a, uint32_t *b, uint32_t na, uint32_t nb, uint32_t* result_size) {
#if COUNT_TIME
        auto  t1 = read_timer();
#endif
    __shared__ uint32_t out_offset[THREADS_PER_BLOCK];
    __shared__ uint32_t out_size;
    
    int lid = threadIdx.x;
    
    if (lid == 0)
        out_size = 0;

    uint32_t num_done = 0;
    while (num_done < na) {
        bool found = 0;
        uint32_t u = 0;
        if (num_done + lid < na) {
            int mid, l = 0, r = nb - 1; // [l, r], use signed int instead of unsigned int!
            u = a[num_done + lid]; // u: an element in set a
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
        __syncthreads();

        for (int s = 1; s < THREADS_PER_BLOCK; s *= 2) {
            uint32_t v = lid >= s ? out_offset[lid - s] : 0;
            __syncthreads();
            out_offset[lid] += v;
            __syncthreads();
        }

        if (found) {
            uint32_t offset = out_offset[lid] - 1;
            out[out_size + offset] = u;
        }

        if (lid == 0)
            out_size += out_offset[THREADS_PER_BLOCK - 1];
        num_done += THREADS_PER_BLOCK;
    }
    __syncthreads();
    if( lid == 0)
        *result_size = out_size;
#if COUNT_TIME
    auto t3 = read_timer();
    if (threadIdx.x % 32 == 0) {
        atomicAdd((unsigned long long*)&i_time, t3 - t1);
    }
#endif
    
}

__device__ void detect_v2(uint32_t *tmp, uint32_t size, uint32_t v2, bool *p_hit) {
    uint32_t i = 0;
    bool hit = false;
    while(i + threadIdx.x < size) {
        hit |= tmp[i+threadIdx.x] == v2;
        i += THREADS_PER_BLOCK;
    }
    if(hit) *p_hit = true;
}

__device__ void upd_ans(uint32_t *a, uint32_t *b, uint32_t na, uint32_t nb, unsigned long long *p_sum) {
    __shared__ uint32_t out_size[32];
    
    int lid = threadIdx.x;
#if COUNT_INTERSECTION
    if(lid == 0) {
        atomicAdd((unsigned long long*)&i_inter_size_1, (uint64_t)na + nb);
        atomicAdd((unsigned long long*)&i_inter_time_1, (uint64_t)1);
    }
#endif
    
    out_size[lid] = 0;

    uint32_t num_done = 0;
    while (num_done < na) {
        bool found = 0;
        uint32_t u = 0;
        if (num_done + lid < na) {
            int mid, l = 0, r = nb - 1; // [l, r], use signed int instead of unsigned int!
            u = a[num_done + lid]; // u: an element in set a
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
        out_size[lid] += found;
        num_done += THREADS_PER_BLOCK;
    }
   __syncthreads();

    for (int s = 1; s < THREADS_PER_BLOCK; s *= 2) {
        uint32_t v = lid >= s ? out_size[lid - s] : 0;
        __syncthreads();
        out_size[lid] += v;
        __syncthreads();
    }
    if(lid == 0)
        *p_sum -= out_size[31];

    /*   __shared__ uint32_t lblock[THREADS_PER_BLOCK];
         __shared__ uint32_t rblock[THREADS_PER_BLOCK];

         uint32_t i = 0, j = 0;
         unsigned long long sum = 0;
         uint32_t lsize = THREADS_PER_BLOCK, rsize = THREADS_PER_BLOCK;

         while (i < ln && j < rn) {

         lsize = min(ln - i, THREADS_PER_BLOCK);
         rsize = min(rn - j, THREADS_PER_BLOCK);

         if(i + threadIdx.x < ln) lblock[threadIdx.x] = lbases[i + threadIdx.x];
         if(j + threadIdx.x < rn) rblock[threadIdx.x] = rbases[j + threadIdx.x];

         __threadfence_block();

         for(int k = 0; k < rsize; ++k)
         sum += (threadIdx.x < lsize) & (lblock[threadIdx.x] == rblock[k]);

         uint32_t llast = lblock[lsize - 1];
         uint32_t rlast = rblock[rsize - 1];

         if(llast >= rlast) j += rsize;
         if(llast <= rlast) i += lsize;
         }

         (*p_sum) -= sum;*/
    /*    i = 0;
          j = 0;
          unsigned long long size = 0;
          while(i < ln && j < rn) {
          if(lbases[i]==rbases[j] && i % 32 == threadIdx.x) {
          ++size;
          }
          int u = lbases[i],v=rbases[j];
          i+=u<=v;
          j+=v<=u;
          }
          assert(size==sum);*/

}

__global__ void __dfs(uint32_t edge_num, uint32_t buffer_size, uint32_t *edge_from, uint32_t *edge, uint32_t *vertex, uint32_t *tmp) {
    __shared__ unsigned int edgeI;
    __shared__ unsigned int edgeEnd;
    __shared__ unsigned long long sdata[THREADS_PER_BLOCK];

    unsigned long long mysum = 0;

    uint32_t tmp1_begin = buffer_size * 2 * blockIdx.x;
    uint32_t tmp2_begin = tmp1_begin + buffer_size;

    if(threadIdx.x == 0) {
        edgeI = edgeEnd = 0;
    }

    __syncthreads();

    assert( edgeI == edgeEnd);

    uint32_t v0,v1,v2;
    __shared__ uint32_t tmp1_size;
    __shared__ uint32_t tmp2_size;

    uint32_t lb,le,ln;
    uint32_t rb,re,rn;

    uint32_t loop_begin, loop_limit;

    __shared__ bool have_v2;

#if COUNT_INTERSECTION
    int lid = threadIdx.x;
#endif
#if COUNT_TIME
    bool do_work = false;
#endif

    while(true) {
        if(threadIdx.x == 0) {
            //printf("%d at %u\n", blockIdx.x, edgeI);
            if(++edgeI >= edgeEnd) {
                edgeI = atomicAdd(&dev_nowEdge, 1);
                edgeEnd = min(edge_num, edgeI + 1);
            }
        }

        __syncthreads();

        unsigned int i = edgeI;
        if(i >= edge_num) break;
#if COUNT_TIME
        do_work = true;
#endif

        // for edge in E
        v0 = edge_from[i];
        v1 = edge[i];

        if(v0 < v1) continue; // optimize

        lb = vertex[v0];
        le = vertex[v0+1];
        ln = le - lb;

        rb = vertex[v1];
        re = vertex[v1+1];
        rn = re - rb;
#if COUNT_INTERSECTION
    if(lid == 0) {
        atomicAdd((unsigned long long*)&i_inter_size_0, (uint64_t)ln + rn);
        atomicAdd((unsigned long long*)&i_inter_time_0, (uint64_t)1);
    }
#endif

        intersection(tmp + tmp1_begin, edge + lb, edge + rb, ln, rn, &tmp1_size); // v3's set = tmp1 = N(v0) & N(v1)
        __syncthreads();
        
        if(tmp1_size == 0) continue;

        loop_begin = vertex[v0];
        loop_limit = vertex[v0+1];
        for(uint32_t j = loop_begin; j < loop_limit; ++j) {
            v2 = edge[j]; // for v2 in N(v1)
            if(v1==v2) continue;

            lb = vertex[v2];
            le = vertex[v2+1];
            ln = le - lb;
#if COUNT_INTERSECTION
    if(lid == 0) {
        atomicAdd((unsigned long long*)&i_inter_size_2, (uint64_t)ln + rn);
        atomicAdd((unsigned long long*)&i_inter_time_2, (uint64_t)1);
    }
#endif

            intersection(tmp + tmp2_begin, edge + lb, edge + rb, ln, rn, &tmp2_size); // v4's set = tmp2 = N(v0) & N(v2)
            __syncthreads();

            if(tmp2_size <= 1) continue;

            if(threadIdx.x == 0) {
                have_v2 = false;
            }
            __syncthreads();

            detect_v2(tmp + tmp1_begin, tmp1_size, v2, &have_v2); // notice that v2 may belong to tmp1, but we want tmp1 - {v2}

            upd_ans(tmp + tmp1_begin, tmp + tmp2_begin, tmp1_size, tmp2_size, &mysum); // ans -= (tmp1 & tmp2).size
            __syncthreads();

            if(threadIdx.x == 0) { // ans += tmp1.size * tmp2.size, notice that v1 always exist in tmp2, so we use tmp2_size-1
                if(have_v2) mysum += (tmp1_size - 1) * (tmp2_size - 1);
                else mysum += tmp1_size * (tmp2_size - 1);
            }
        }
    }

    sdata[threadIdx.x] = mysum;
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
#if COUNT_TIME
        if(do_work) atomicAdd(&i_count, 1);
#endif
    }
    
}

void gpu_pattern_matching(Graph *g) {
    uint32_t *edge_from = new uint32_t[g->e_cnt];
    for(uint32_t i = 0; i < g->v_cnt; ++i)
        for(uint32_t j = g->vertex[i]; j < g->vertex[i+1]; ++j)
            edge_from[j] = i;

    uint32_t *edge = new uint32_t[g->e_cnt];
    uint32_t *vertex = new uint32_t[g->v_cnt + 1];

    for(uint32_t i = 0;i < g->e_cnt; ++i) edge[i] = g->edge[i];
    for(uint32_t i = 0;i <= g->v_cnt; ++i) vertex[i] = g->vertex[i];

    tmpTime.check(); 
    int numBlocks = 4096;

    uint32_t size_edge = g->e_cnt * sizeof(uint32_t);
    uint32_t size_vertex = (g->v_cnt + 1) * sizeof(uint32_t);
    uint32_t size_tmp  = VertexSet::max_intersection_size * sizeof(uint32_t) * numBlocks * (1 + 1);

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

    tmpTime.print("Prepare time cost");
    tmpTime.check();

    uint32_t edge_num = g->e_cnt;
    uint32_t buffer_size = VertexSet::max_intersection_size;
    __dfs<<<numBlocks, THREADS_PER_BLOCK>>>(edge_num, buffer_size, dev_edge_from, dev_edge, dev_vertex, dev_tmp);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk( cudaMemcpyFromSymbol(&sum, dev_sum, sizeof(sum)) );

#if COUNT_INTERSECTION
    uint64_t i_inter_size_cpu_0, i_inter_time_cpu_0;
    gpuErrchk( cudaMemcpyFromSymbol(&i_inter_size_cpu_0, i_inter_size_0, sizeof(i_inter_size_0)));
    gpuErrchk( cudaMemcpyFromSymbol(&i_inter_time_cpu_0, i_inter_time_0, sizeof(i_inter_time_0)));
    printf("inter time0 %llu, inter size0 %llu \n", i_inter_time_cpu_0, i_inter_size_cpu_0);
    
    uint64_t i_inter_size_cpu_1, i_inter_time_cpu_1;
    gpuErrchk( cudaMemcpyFromSymbol(&i_inter_size_cpu_1, i_inter_size_1, sizeof(i_inter_size_1)));
    gpuErrchk( cudaMemcpyFromSymbol(&i_inter_time_cpu_1, i_inter_time_1, sizeof(i_inter_time_1)));
    printf("inter time1 %llu, inter size1 %llu \n", i_inter_time_cpu_1, i_inter_size_cpu_1);
    
    uint64_t i_inter_size_cpu_2, i_inter_time_cpu_2;
    gpuErrchk( cudaMemcpyFromSymbol(&i_inter_size_cpu_2, i_inter_size_2, sizeof(i_inter_size_2)));
    gpuErrchk( cudaMemcpyFromSymbol(&i_inter_time_cpu_2, i_inter_time_2, sizeof(i_inter_time_2)));
    printf("inter time2 %llu, inter size2 %llu \n", i_inter_time_cpu_2, i_inter_size_cpu_2);
#endif

#if COUNT_TIME
    int i_count_cpu;
    uint64_t t_time_cpu, i_time_cpu, ihead_time_cpu;
    gpuErrchk( cudaMemcpyFromSymbol(&i_count_cpu, i_count, sizeof(i_count)));
    gpuErrchk( cudaMemcpyFromSymbol(&i_time_cpu, i_time, sizeof(i_time)));
    printf("Warp count: %d\nIntersection time: %ld\n", i_count_cpu, i_time_cpu);
#endif

    printf("house count %llu\n", sum);
    tmpTime.print("Counting time cost");
}

int main(int argc,char *argv[]) {
    Graph *g;
    DataLoader D;

    /*
    const std::string type = argv[1];
    const std::string path = argv[2];

    DataType my_type;

    GetDataType(my_type, type);

    if(my_type == DataType::Invalid) {
        printf("Dataset not found!\n");
        return 0;
    }

    D.load_data(g,my_type,path.c_str());
    //assert(D.load_data(g,my_type,path.c_str())==true); 

 //   assert(D.load_data(g,10));
    printf("Load data success!\n");
    fflush(stdout);
    */

    bool ok = D.fast_load(g, argv[1]);

    if (!ok) {
        printf("data load failure :-(\n");
        return 0;
    }

    allTime.check();

    gpu_pattern_matching(g);

    allTime.print("Total time cost");

    return 0;
}
