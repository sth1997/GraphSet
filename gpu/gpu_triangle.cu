// This program can only count triangle using GPU.
#include <assert.h>
#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <algorithm>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

# include <sys/time.h>
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
TimeInterval preProcessTime;
TimeInterval tmpTime;
    
uint32_t N,edge_num;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__device__ unsigned long long dev_sum;
__device__ unsigned int dev_nowNode;

__device__ void intersection(const uint32_t* lbases, const uint32_t* rbases, uint32_t ln, uint32_t rn, unsigned long long* p_mysum) {
    if( ln == 0 || rn == 0) return;
    
    uint32_t i = 0, sum = 0;
    uint32_t size;

    uint32_t my_id;

    uint32_t s_l,s_r,s_mid;
    while( i < ln ) {
      //  if(threadIdx.x==0) printf("loop %u\n", i);
        size = min(ln-i, 32);
        if( threadIdx.x < size) {
            my_id = lbases[threadIdx.x + i];

            s_l = 0;
            s_r = rn;
            // You CANNOT write  s_l < s_r - 1 because type is UNSIGNED
            while(s_l + 1 < s_r ) {
              //  if(threadIdx.x == 0) printf("binary %u %u\n", s_l, s_r);
                s_mid = (s_l + s_r) >> 1;
                if(my_id < rbases[s_mid]) s_r = s_mid;
                else s_l = s_mid;
            }
            if(my_id == rbases[s_l]) ++sum;

            bool hit = false;
            for(int j = 0; j < rn; ++j)
                if(lbases[threadIdx.x + i] == rbases[j]) {
                    hit = true;
                    break;
                }
            bool Ass = (my_id == rbases[s_l]) == hit;
            if(!Ass) {
                printf("%u %u %d %u\n", ln, rn, threadIdx.x, my_id);
                for(int j = 0; j < rn; ++j)
                    printf("%u ", rbases[j]);
                printf("\n");
            }
            assert((my_id==rbases[s_l]) == hit );
        }
        i += size;
    }

    (*p_mysum) += sum;
}

__global__ void __tricount(uint32_t N, const uint32_t* __restrict__ vertex, const uint32_t* __restrict__ edge) {
    __shared__ unsigned int nodeI;
    __shared__ unsigned int nodeEnd;
    __shared__ unsigned long long sdata[32];

    unsigned long long mysum = 0;
    
    if(threadIdx.x == 0) {
        nodeI = nodeEnd = 0;
    }

    __syncthreads();
    
    assert( nodeI == nodeEnd);

    while(true) {
        if(threadIdx.x == 0) {
            if(++nodeI >= nodeEnd) {
                nodeI = atomicAdd(&dev_nowNode, 256);
                nodeEnd = min(N, nodeI + 256);

//                if(nodeI < N) printf("%d at %u\n", blockIdx.x, nodeI);
            }
        }

        __syncthreads();

        unsigned int i = nodeI;

        if(i >= N) break;

//        if(threadIdx.x==0) printf("%d %d %u achieve here\n", blockIdx.x, threadIdx.x, nodeI);

        uint32_t lb = vertex[i];
        uint32_t le = vertex[i+1];
        uint32_t ln = le - lb;

        for(uint32_t j = lb; j < le; ++j) {
            uint32_t ri = edge[j];
            uint32_t rn = vertex[ri+1] - vertex[ri];
            uint32_t rb = vertex[ri];

            assert(ln>=0);
            assert(rn>=0);
           // assert(lb<vertex[N]);
           // assert(rb<vertex[N]);
            intersection(edge + lb, edge + rb, ln, rn, &mysum);
        }
    }
    
    sdata[threadIdx.x] = mysum;
    __syncthreads();

   // if(threadIdx.x == 0) printf("%d achieve here\n", blockIdx.x);

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

bool cmp_pair(std::pair<int,int>a, std::pair<int,int>b) {
    return a.first < b.first || (a.first == b.first && a.second < b.second);
}

bool load_data(uint32_t* &vertex, uint32_t* &edge, const char* path) {
    if (freopen(path, "r", stdin) == NULL)
    {
        printf("File not found. %s\n", path);
        return false;
    }
    printf("Load begin in %s\n",path);

    scanf("%u%u",&N,&edge_num);
    
    std::pair<int,int> *e = new std::pair<int,int>[edge_num];
    std::map<int,int> id;
    id.clear();
    int x,y;
    int tmp_v;
    unsigned int tmp_e;
    tmp_v = 0;
    tmp_e = 0;
    while(scanf("%d%d",&x,&y)!=EOF) {
        if(x == y) {
            printf("find self circle\n");
            edge_num -=1;
            continue;
            //return false;
        }
        if(!id.count(x)) id[x] = tmp_v ++;
        if(!id.count(y)) id[y] = tmp_v ++;
        
        x = id[x];
        y = id[y];
        if( x > y ) std::swap(x,y);
        e[tmp_e++] = std::make_pair(x,y);
        
        if(tmp_e % 1000000u == 0u) {
            printf("load %u edges\n",tmp_e);
            fflush(stdout);
        }
    }

    if(tmp_v != N) {
        printf("vertex number error!\n");
    }
    if(tmp_e != edge_num) {
        printf("edge number error!\n");
    }
    if(tmp_v != N || tmp_e != edge_num) {
        fclose(stdin);
        delete[] e;
        return false;
    }
    
    std::sort(e,e+tmp_e,cmp_pair);
    edge_num = unique(e,e+tmp_e) - e;
    for(unsigned int i = 0; i < edge_num - 1; ++i)
        if(e[i] == e[i+1]) {
            printf("have same edge\n");
            fclose(stdin);
            delete[] e;
            return false;
        }

    vertex = new uint32_t[N+1];
    edge = new uint32_t[edge_num];

    int cur_node = -1;

    for(uint32_t i = 0; i < edge_num; ++i){
       while( cur_node < e[i].first ) vertex[++cur_node] = i;
       edge[i] = e[i].second;
    }
    while( cur_node < N ) vertex[++cur_node] = edge_num;

    for(uint32_t i = 0; i < N; ++i)
        assert(vertex[i]<=vertex[i+1]);
    assert(vertex[N]==edge_num);

    return true;
}

void triangle_counting(uint32_t *vertex, uint32_t *edge) {
    
    int numBlocks = 2048;
    
    uint32_t *dev_vertex;
    uint32_t *dev_edge;

    uint64_t size_vertex = sizeof(uint32_t) * (N + 1);
    uint64_t size_edge = sizeof(uint32_t) * edge_num;

    tmpTime.check();

    gpuErrchk( cudaMalloc((void**)&dev_vertex, size_vertex));
    gpuErrchk( cudaMalloc((void**)&dev_edge, size_edge));

    gpuErrchk( cudaMemcpy(dev_vertex, vertex, size_vertex, cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy(dev_edge, edge, size_edge, cudaMemcpyHostToDevice));

    unsigned long long sum = 0;
    gpuErrchk( cudaMemcpyToSymbol(dev_sum, &sum, sizeof(unsigned long long), 0, cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpyToSymbol(dev_nowNode, &sum, sizeof(unsigned int), 0, cudaMemcpyHostToDevice));
    tmpTime.print("HostToDevice time cost");

    tmpTime.check();
    printf("%u %u\n", N, edge_num);
    printf("achieve here\n"); fflush(stdout);
    __tricount<<<numBlocks, 32>>>(N, dev_vertex, dev_edge);
    
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk( cudaMemcpyFromSymbol(&sum, dev_sum, sizeof(sum)) );
    tmpTime.print("Counting time cost");

    printf("triangle count : %llu\n", sum);
}

int main(int argc,char *argv[]) {
    allTime.check();

    uint32_t *vertex, *edge;
    
    tmpTime.check();

    assert(load_data(vertex, edge, argv[1]));

    tmpTime.print("Load time cost");

    triangle_counting(vertex, edge);

    delete[] vertex;
    delete[] edge;

    allTime.print("Total time cost");

    return 0;
}
