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

__device__ void intersection(uint32_t *tmp, uint32_t *lbases, uint32_t *rbases, uint32_t ln, uint32_t rn, uint32_t* p_tmp_size) {
    __shared__ uint32_t lblock[THREADS_PER_BLOCK];
    __shared__ uint32_t rblock[THREADS_PER_BLOCK];

    __shared__ uint32_t cur_thread;

    uint32_t i = 0, j = 0;
    uint32_t lsize = THREADS_PER_BLOCK, rsize = THREADS_PER_BLOCK;

    bool hit;

    if( threadIdx.x == 0 ) {
        *p_tmp_size = 0;
    }
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

        while( cur_thread < THREADS_PER_BLOCK) {
            if(cur_thread == threadIdx.x) {
                if(hit && i + threadIdx.x < ln) tmp[(*p_tmp_size)++] = lblock[threadIdx.x];
                ++cur_thread;
            }
            __syncthreads();
        }
        
        uint32_t llast = lblock[lsize - 1];
        uint32_t rlast = rblock[rsize - 1];

        if(llast >= rlast) j += rsize;
        if(llast <= rlast) i += lsize;
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

__device__ void detect_v2(uint32_t *tmp, uint32_t size, uint32_t v2, bool *p_hit) {
    uint32_t i = 0;
    bool hit = false;
    while(i + threadIdx.x < size) {
        hit |= tmp[i+threadIdx.x] == v2;
        i += THREADS_PER_BLOCK;
    }
    if(hit) *p_hit = true;
}

__device__ void upd_ans(uint32_t *lbases, uint32_t *rbases, uint32_t ln, uint32_t rn, unsigned long long *p_sum) {
    __shared__ uint32_t lblock[THREADS_PER_BLOCK];
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

    (*p_sum) -= sum;
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
        
        v0 = edge_from[i];
        v1 = edge[i];

        if(v0 > v1) continue; // optimize

        lb = vertex[v0];
        le = vertex[v0+1];
        ln = le - lb;

        rb = vertex[v1];
        re = vertex[v1+1];
        rn = re - rb;

        intersection(tmp + tmp1_begin, edge + lb, edge + rb, ln, rn, &tmp1_size);
        __syncthreads();
        
        if(tmp1_size == 0) continue;

        loop_begin = vertex[v1];
        loop_limit = vertex[v1+1];
        for(uint32_t j = loop_begin; j < loop_limit; ++j) {
            v2 = edge[j];
            if(v0==v2) continue;

            rb = vertex[v2];
            re = vertex[v2+1];
            rn = re - rb;

            intersection(tmp + tmp2_begin, edge + lb, edge + rb, ln, rn, &tmp2_size);
            __syncthreads();

            if(tmp2_size <= 1) continue;

            if(threadIdx.x == 0) {
                have_v2 = false;
            }
            __syncthreads();

            detect_v2(tmp + tmp1_begin, tmp1_size, v2, &have_v2);

            upd_ans(tmp + tmp1_begin, tmp + tmp2_begin, tmp1_size, tmp2_size, &mysum);
            __syncthreads();

            if(threadIdx.x == 0) {
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

    printf("house count %llu\n", sum);
    tmpTime.print("Counting time cost");
}

int main(int argc,char *argv[]) {
    Graph *g;
    DataLoader D;

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

    gpu_pattern_matching(g);

    allTime.print("Total time cost");

    return 0;
}
