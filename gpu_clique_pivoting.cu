#include "graph.h"
#include "dataloader.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <ctime>
#include <iostream>
#include <algorithm>
#include <vector>
#include <chrono>

namespace Pivoting {
#define WATCHING(u) (true)
// #define depot do { if (WATCHING(blockIdx.x)) printf("Device %d at line %d, in '%s'\n", blockIdx.x, __LINE__, __FUNCTION__); } while(0);
#define depot ;
typedef uint64_t bin_t;
#define WIDTH 64
// typedef uint8_t bin_t;
// #define WIDTH 8
#define blocks(n) (((n) - 1) / WIDTH + 1)
#define blockid(x) ((x) / WIDTH)
#define bitid(x) ((x) % WIDTH)
#define flatid(x,y) ((x)*WIDTH+(y))
#define eprintf(...) fprintf(stderr, __VA_ARGS__)
#define pass printf("%3d: in function '%s'\n", __LINE__, __FUNCTION__);
#define sassert(cond) do { if (!(cond)) printf("%d: error\n", __LINE__); } while(0)
#define BEGIN_LEADER do { if (threadIdx.x == 0) {
#define END_LEADER     } else {} __syncthreads(); } while (0);
#define INLINE_LEADER if (threadIdx.x == 0)
// #define BEGIN_LEADER
// #define END_LEADER
// #define INLINE_LEADER
// #define __syncthreads() do{}while(0)
// #define __syncthreads() __syncwwarp()
#define MAXC 998244353 // max clique, 20 for small graphs
#define MAXD 998244353 // max degree 35455, 1100 for small graphs
// should be MAXD ~ 1400
// total shared memory for each thread block
// constexpr int total_shared = MAXC * (5 * 4 + 2 * blocks(MAXD) * sizeof(bin_t));

const int MAX_NTHREADS = 128;
struct Config {
  int blockdim = 32;
  bool valid() const {
    return __builtin_popcount(blockdim) == 1 && blockdim <= MAX_NTHREADS;
  }
} config;



struct Subgraph {
  int v_cnt;
  int64_t e_cnt;
  bin_t **adj;
  Subgraph () {
    v_cnt = -1;
    adj = nullptr;
  }
  // int64_t edges() const;

  // max k that k * (k-1) <= e_cnt
  __host__ __device__ int bound_clique() const {
    int64_t L = 1, R = 1e9; // hope
    while (L < R) {
      int64_t M = (L + R + 1) / 2;
      if (M * (M - 1) <= e_cnt) L = M;
      else R = M - 1;
    }
    return L;
  }
  __host__ __device__ uint32_t dfs_stack_size() const {
    int C = min(bound_clique() + 1, MAXC);
    int n = v_cnt;
    return sizeof(int) * C * 4 + sizeof(bin_t) * C * blocks(n) * 2;
  }
};

int find_pivot(Graph *g) {
  int candidate = 0;
  for (int u = 0; u < g->v_cnt; ++u) {
    if (g->vertex[u + 1] - g->vertex[u] > g->vertex[candidate + 1] - g->vertex[candidate]) {
      candidate = u;
    }
  }
  return candidate;
}

__device__ int64_t binom(int64_t n, int64_t k) {
  if (!(n >= 0 && k <= n && k >= 0)) printf("Cannot compute C(%lld, %lld)\n", n, k);
  if (n - k < k) k = n - k;
  if (!(pow(n, k) < 9e18)) printf("C(%lld, %lld) is too large\n", n, k);
  int64_t result = 1;
  for (int i = 0; i < k; ++i) {
    result = result * (n - i) / (i + 1);
  }
  // printf("C(%lld,%lld)=%lld\n", n, k, result);
  return result;
}

template<typename T>
__device__ int popcount(T x) {
  int r = 0;
  while (x) { r++; x &= (x-1); }
  return r;
}
template<>
__device__ int popcount(unsigned int x) {
  return __popc(x);
}
template<>
__device__ int popcount(uint64_t x) {
  return __popcll(x);
}

template<typename T>
__device__ int popcount(T* x, int n) {
  __shared__ int r[MAX_NTHREADS];
  int tid = threadIdx.x;
  r[tid] = 0;
  for (int i = tid; i < blocks(n); i += blockDim.x) r[tid] += popcount(x[i]);
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    __syncthreads();
    if (tid < s) r[tid] += r[tid + s];
  }
  __syncthreads();
  return r[0];
}
// block i: calc() -> threads
template<typename T>
__device__ int popcount_intersection(T* I, T* J, int n) {
  __shared__ int r[MAX_NTHREADS];
  int tid = threadIdx.x;
  r[tid] = 0;
  for (int i = tid; i < blocks(n); i += blockDim.x) r[tid] += popcount(I[i] & J[i]);
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    __syncthreads();
    if (tid < s) r[tid] += r[tid + s];
  }
  __syncthreads();
  return r[0];
}

// __host__ int64_t Subgraph::edges() const {
//   int n = this->v_cnt;
//   int64_t m = 0;
//   for (int i = 0; i < n; ++i)
//     for (int j = 0; j < blocks(n); ++j)
//       m += __builtin_popcount(this->adj[i][j]);
//   return m;
// }

__device__ void intersection(bin_t *res, bin_t *I, bin_t *J, int n) {
  for (int i = threadIdx.x; i < blocks(n); i += blockDim.x) res[i] = I[i] & J[i];
}
__device__ void setminus(bin_t *res, bin_t *I, bin_t *J, int n) {
  for (int i = threadIdx.x; i < blocks(n); i += blockDim.x) res[i] = I[i] & (~J[i]);
}
// res = I - {x in J | x < k}
// note: this should be inplace
__device__ void setminus_smaller(bin_t *res, bin_t *I, int k, int n) {
  sassert(0 <= k && k < n);
  if (!(0 <= k && k < n)) return;
  for (int i = threadIdx.x; i <= blockid(k); i += blockDim.x) {
    if (i < blockid(k)) res[i] &= ~I[i];
    else res[blockid(k)] &= (~I[blockid(k)] | (~0ull << bitid(k)));
  }
}

#define spa do { if (blockIdx.x == 323) printf("block %d on line %d\n", blockIdx.x, __LINE__);  } while (0);

__device__ int find_pivot_gpu(bin_t *I, Subgraph *G) {
  // printf("%s: I=%llx\n", __FUNCTION__, I[0]);
  __shared__ int cand, cand_deg;
  cand = -1; cand_deg = -1;
  __syncthreads();
  for (int i = 0; i < blocks(G->v_cnt); ++i) if (I[i]) {
    for (int j = 0; j < WIDTH; ++j) if ((I[i] >> j) & 1) {
      int u = flatid(i, j);
      if (u >= G->v_cnt) printf("(%d, %d) uses %d out of v_cnt = %d\n", blockIdx.x, threadIdx.x, u, G->v_cnt);
      sassert(u < G->v_cnt);
      int deg = popcount_intersection(I, G->adj[flatid(i, j)], G->v_cnt);
      // printf("find_pivot_gpu: inter(%d) = %d\n", u, deg);
      __syncthreads();
      if (threadIdx.x == 0 && deg > cand_deg) cand = u, cand_deg = deg;
      __syncthreads();
    }
  }
  __syncthreads();
  sassert(cand != -1);
  if (cand == -1) printf("(%d, %d) uses %d out of v_cnt = %d, with I[0] = %lld, cnt = %d\n", blockIdx.x, threadIdx.x, cand, G->v_cnt, I[0], popcount(I, G->v_cnt));
  return cand;
}

__device__ int64_t pivoting_on_extracted_block(Subgraph *G, int k, int l_big, bin_t *I_big, int n_pivots_big, char* pool) {
  int top, n = G->v_cnt;
  // if (n == 0 || (I_big && popcount(I_big, n) == 0)) return 0; // n == 1 indicates this subtree has one node only
  // if (!WATCHING(blockIdx.x) || n > 150) return 0;
  if (n > MAXD) {
    BEGIN_LEADER
      printf("Error: n = %d > MAXD = %d\n", n, MAXD);
    END_LEADER
    return 0;
  }
  if (n == 0) return 0;
  __shared__ bin_t *I, *Ipruned;
  __shared__ int * l, * npivots, * vpivot, * inow;

  __shared__ int64_t num_cliques;
  __shared__ int C;
  
  top = 0;
  BEGIN_LEADER
    C = min(G->bound_clique() + 1, MAXC);
    I = (bin_t*)pool;
    Ipruned = (bin_t*)&I[C*blocks(n)];
    l = (int*)&Ipruned[C*blocks(n)];
    npivots = (int*)&l[C];
    vpivot = (int*)&npivots[C];
    inow = (int*)&vpivot[C];
    sassert((char*)&inow[C] - pool == G->dfs_stack_size());

    num_cliques = 0;
    l[top] = l_big;
    memset(I, 0, sizeof(bin_t) * blocks(n));
    // if (I_big) memcpy(I, I_big, sizeof(bin_t) * blocks(n));
    // else
    for (int i = 0; i < n; ++i) I[top*blocks(n) + blockid(i)] |= (1ull << bitid(i));
    npivots[top] = n_pivots_big;
  END_LEADER
  {
    int cnt = popcount(I, n);
    if (cnt == 0) return 0;
  }
  vpivot[top] = find_pivot_gpu(I, G); // set should be okay
  if (vpivot[top] == -1) return 0;
  __syncthreads();
  // printf("subgraph: n = %d, vpivot = %d, top = %d\n", n, vpivot[top], top);
  setminus(Ipruned, I, G->adj[vpivot[top]], n);
  // printf("Ipruned = %lld\n", Ipruned[top][0]);
  BEGIN_LEADER
    npivots[top] = n_pivots_big;
    inow[top] = -1;
  END_LEADER
  int cnter = 0;
  while (top >= 0) {
    if (threadIdx.x == 0 && ((++cnter) & 8388607) == 0) printf("(block %4d): vc = %4d, inow[0] = %4d, inow[1] = %4d\n", blockIdx.x, G->v_cnt, inow[0], top >= 1 ? inow[1] : -1);
    // if (blockIdx.x == 0 && threadIdx.x == 0 && top == 0) printf("top = %d, inow = %d\n", top, inow[top]);
    // if (threadIdx.x == 0 && top == 0 && inow[top] > 1000) printf("(block %d): top = %d, inow = %d\n", gridDim.x, top, inow[top]);
    if (!(top+1 < C)) {
      INLINE_LEADER printf("block %d reports top=%d, vcnt=%d, ecnt=%lld\n", blockIdx.x, top, G->v_cnt, G->e_cnt);
      return 0;
    }
    #define i inow[top]
    BEGIN_LEADER
      do { // can be accelerated
        ++i;
      } while (i < n && ((Ipruned[top*blocks(n)+blockid(i)] >> bitid(i)) & 1) == 0);
    END_LEADER
    // printf("i = %d\n", i);
    __syncthreads();
    if (i == n) { --top; __syncthreads(); continue; }

    INLINE_LEADER npivots[top + 1] = (i == vpivot[top]) ? (npivots[top] + 1) : npivots[top];
    __syncthreads();
    // printf("top = %d, node = %d (pivot=%d), npivots = %d\n", top, i, vpivot[top], npivots[top+1]);
    if (l[top] + 1 - k <= npivots[top + 1]) {
      intersection(&I[(top+1)*blocks(n)], &I[top*blocks(n)], G->adj[i], n);
      setminus_smaller(&I[(top+1)*blocks(n)], &Ipruned[top*blocks(n)], i, n);
    #undef i
      // printf("I' = %lld\n", I[top+1][0]);
      int cnt = popcount(&I[(top+1)*blocks(n)], n);
      if (cnt > 0) {
        // vpivot, Ipruned, inow
        INLINE_LEADER l[top + 1] = l[top] + 1;
        __syncthreads();
        // I[top+1], npivots[top+1] already set
        vpivot[top+1] = find_pivot_gpu(&I[(top+1)*blocks(n)], G);
        INLINE_LEADER sassert(vpivot[top+1] != -1);
        setminus(&Ipruned[(top+1)*blocks(n)], &I[(top+1)*blocks(n)], G->adj[vpivot[top+1]], n);
        BEGIN_LEADER
          inow[top+1] = -1;
          // printf("push(%d): pivot = %d, Ipruned = %lld\n", top+1, vpivot[top+1], Ipruned[top+1][0]);
        END_LEADER
        top++;
      } else if (l[top]+1 >= k)
        INLINE_LEADER num_cliques += binom(npivots[top+1], l[top] + 1 - k);
    }
    __syncthreads();
  }
  return num_cliques;
}

__global__ void prepare_gpu(bin_t *h_adj_pool, bin_t **h_ptr_pool, Subgraph *h_g_pool,
  Subgraph **h_gs, int64_t *h_degacc, int64_t *h_deg2acc) {
  sassert(blockDim.x == 1);
  int u = blockIdx.x;
    h_gs[u] = h_g_pool + u;
    h_gs[u]->adj = h_ptr_pool + h_degacc[u];
    for (int i = 0; i < h_gs[u]->v_cnt; ++i)
      h_gs[u]->adj[i] = h_adj_pool + h_deg2acc[u] + i*blocks(h_gs[u]->v_cnt);
  int n = h_gs[u]->v_cnt;

  if (h_gs[u]->v_cnt > MAXD) {
      printf("Wrong config: block %d has n = %d > MAXD = %d\n", blockIdx.x, n, MAXD);
    return;
  }
}

__global__ void pivoting_traverse(Subgraph **h_gs, int k, int64_t *ans, char* dfs_pool, int64_t* dfs_offset) { // pivot is vertex 0
  int u = blockIdx.x;
  // INLINE_LEADER {
  //   sassert(sizeof(pool) >= h_gs[u]->dfs_stack_size());
  // }
  // if (!(sizeof(pool) >= h_gs[u]->dfs_stack_size())) {
  //   printf("Error\n"); return;
  // }
  // __syncthreads();
  int64_t numcliques = pivoting_on_extracted_block(h_gs[u], k, 1, nullptr, u == 0 ? 1 : 0, dfs_pool + dfs_offset[u]);
  BEGIN_LEADER
    ans[blockIdx.x] = numcliques;
  END_LEADER
}

struct Timer {
  #define duration(a) std::chrono::duration_cast<std::chrono::milliseconds>(a).count()
  #define timeNow() std::chrono::high_resolution_clock::now()
  typedef std::chrono::high_resolution_clock::time_point TimeVar;
  TimeVar st, et;
  Timer() {
    reset();
  }
  void reset() {
    st = timeNow();
  }
  int64_t gettime() {
    return duration(et - st);
  }
  void report(std::string s) {
    et = timeNow();
    int64_t x = duration(et - st);
    eprintf("Timer '%s' counts %lldms\n", s.c_str(), x);
  }
  #undef duration
  #undef timeNow
};

Timer overhead_timer;
Timer exec_timer;

int64_t k_clique_counting_pivoting(Graph* g, int k) {

overhead_timer.reset();
  int n = g->v_cnt;
  int pivot = find_pivot(g);
  eprintf("pivot is %d with deg=%lld\n", pivot, g->vertex[pivot+1]-g->vertex[pivot]);
  int *rid = new int[n];
  memset(rid, -1, sizeof(int)*n);
  // pivot is number zero
  // reorder V-adj(pivot) into 0~(n-|adj(pivot)|)
  // rid can be: -1 for unset, -2 for adj(pivot), 0 for pivot, x>=1 for others
  int u_running = 0;
  rid[pivot] = 0; u_running++;
  for (int i = g->vertex[pivot]; i < g->vertex[pivot+1]; ++i) {
    int v = g->edge[i];
    rid[v] = -2;
  }
  for (int v = 0; v < n; ++v) if (rid[v] != -2 && v != pivot) {
    rid[v] = u_running++;
  }
  eprintf("general_pivot = %d, u_running = %d\n", pivot, u_running);

  // figure out size of subgraphs with vertex u
  // for u==pivot, size(subgraph) = 1+deg(u)
  // for u!=pivot, size(subgraph) = 1+#I'

  int64_t adj_size = 0, ptr_size = 0;
  for (int u = 0; u < n; ++u) {
    int r = rid[u];
    sassert(r != -1);
    if (r == -2) continue;
    int deg = 0;

    // on here we must force
    // rid[x] >= 0   <=>   x in I_pruned
    // I' = adj(u) - {v in Ip  |  v < u}
    for (int i = g->vertex[u], id = 0; i < g->vertex[u + 1]; ++i) {
      int v = g->edge[i];
      sassert(v != u); // no self-loop
      if (rid[v] >= 0 && u > v) continue;
      id++; deg = id;
    }
    adj_size += (deg+1) * blocks(deg+1);
    ptr_size += deg+1;
    // printf("size of %d estimated as %d\n", u, deg+1);
  }

  eprintf("adj_size takes = %lld Bytes, ptr_size takes %lld Bytes\n", adj_size * sizeof(bin_t), ptr_size * sizeof(bin_t*));
  printf("k = %d\n", k);

  /*reserve enough space for pivoting*/
  /*the "n" below can be changed into u_running*/
  bin_t *adj_pool = (bin_t*)malloc(adj_size*sizeof(bin_t));
  memset(adj_pool, 0, adj_size*sizeof(bin_t));
  bin_t **ptr_pool = (bin_t**)malloc(ptr_size*sizeof(bin_t*));
  bin_t *adj_running = adj_pool, **ptr_running = ptr_pool;
  Subgraph *g_pool = (Subgraph*)malloc(n*sizeof(Subgraph));
  Subgraph **gs = (Subgraph**)malloc(n*sizeof(Subgraph*));
  int *uid = (int*)malloc(n*sizeof(int));
  memset(uid, -1, n*sizeof(int));
  int64_t *ptr_offset = (int64_t*)malloc(n * sizeof(int64_t));
  int64_t *adj_offset = (int64_t*)malloc(n * sizeof(int64_t));
  // eprintf("Sufficient CPU memory.\n");
  // eprintf("Shared gpu memory is estimated to be: %d Bytes\n", total_shared);

  for (int u = 0; u < n; ++u) {
    int r = rid[u];
    // eprintf("node %d of id %d\n", u, r);
    sassert(r != -1);
    if (r == -2) continue;
    gs[r] = g_pool + r;
    // on here we must force
    // rid[x] >= 0   <=>   x in I_pruned, I_p = I - adj(pivot)
    // I' = adj(u) - {v in Ip  |  v < u}
    std::vector<int> neib; // neib = {u, adj(u)}
    // uid can be: 0 for u, x>=1 for adj(u), -1 for others
    // uid[neib[i]] == i
    // uid[u] = 0;
    // neib.push_back(u);
    for (int i = g->vertex[u]; i < g->vertex[u + 1]; ++i) {
      int v = g->edge[i];
      sassert(v != u); // no self-loop
      if (rid[v] >= 0 && u > v) continue;
      uid[v] = neib.size();
      neib.push_back(v);
    }
    // printf("Subgraph of #%d (size=%d):", u, (int)neib.size());
    // for (int x : neib) printf(" %d", x);
    // printf("\n");
    gs[r]->v_cnt = (int)neib.size();
    gs[r]->e_cnt = 0;
    ptr_offset[r] = ptr_running - ptr_pool;
    adj_offset[r] = adj_running - adj_pool;
    gs[r]->adj = ptr_running;
    ptr_running += gs[r]->v_cnt;
    for (int i = 0; i < (int)neib.size(); ++i) {
      int v = neib[i];
      sassert(uid[v] == i);
      gs[r]->adj[i] = adj_running;
      adj_running += blocks(gs[r]->v_cnt);
      for (int id = g->vertex[v]; id < g->vertex[v + 1]; ++id) if (uid[g->edge[id]] != -1) {
        int w = g->edge[id];
        int j = uid[w];
        gs[r]->adj[i][blockid(j)] |= (1ull << bitid(j));
        gs[r]->e_cnt++;
      }
    }
    for (int i = 0; i < (int)neib.size(); ++i)
      for (int j = 0; j < (int)neib.size(); ++j)
        sassert((gs[r]->adj[i][blockid(j)] >> bitid(j) & 1) == (gs[r]->adj[j][blockid(i)] >> bitid(i) & 1));
    for (int i = g->vertex[u]; i < g->vertex[u + 1]; ++i) {
      int v = g->edge[i];
      uid[v] = -1;
    }
  }

  eprintf("u_running=%d, ptr_total=%lld, adj_total=%lld\n", u_running, (long long)(ptr_running - ptr_pool), (long long)(adj_running - adj_pool));
  int64_t max_e_cnt = -1;
  int u_of_max_e_cnt = -1;
  for (int i = 0; i < u_running; ++i) if (max_e_cnt < gs[i]->e_cnt) max_e_cnt = gs[i]->e_cnt, u_of_max_e_cnt = i;
  eprintf("Max e_cnt = %lld, by transformed id %d, clique bound = %d, v_cnt = %d\n", max_e_cnt, u_of_max_e_cnt, gs[u_of_max_e_cnt]->bound_clique(), gs[u_of_max_e_cnt]->v_cnt);

  int64_t *dfs_offset = (int64_t*)malloc((u_running+1) * sizeof(int64_t));
  for (int i = 0; i <= u_running; ++i) dfs_offset[i] = 0;
  for (int i = 1; i <= u_running; ++i) dfs_offset[i] = dfs_offset[i - 1] + gs[i - 1]->dfs_stack_size();
  int64_t total_dfs = dfs_offset[u_running];
  eprintf("dfs_pool takes %lld Bytes\n", total_dfs);
  char *dfs_pool;
  cudaMalloc((void**)&dfs_pool, total_dfs);
  int64_t *h_dfs_offset;
  cudaMalloc((void**)&h_dfs_offset, u_running*sizeof(int64_t));
  cudaMemcpy(h_dfs_offset, dfs_offset, sizeof(int64_t) * u_running, cudaMemcpyHostToDevice);

  // for (int u = 0; u < u_running; ++u) if (WATCHING(u)) {
  //     int cntl = 0, cntm = 0, cntr = 0;
  //     int n = gs[u]->v_cnt;
  // for (int i = 0; i < n; ++i) {
  //   for (int j = 0; j < n; ++j) if (gs[u]->adj[i][blockid(j)] >> bitid(j) & 1) {
  //     if (i < j) ++cntl;
  //     else if (i > j) ++cntr;
  //     else ++cntm;
  //   }
  // }
  // printf("Host Master %d: v = %d, ecnt = %d,%d,%d, off=%d,%d\n", u, gs[u]->v_cnt, cntl, cntm, cntr, (int)(gs[u]->adj - ptr_pool), (int)(gs[u]->adj[0] - adj_pool));
  // }

  // h_ prefix indicates gpu
  bin_t *h_adj_pool; cudaMalloc((void**)&h_adj_pool, adj_size*sizeof(bin_t));
  bin_t **h_ptr_pool; cudaMalloc((void**)&h_ptr_pool, ptr_size*sizeof(bin_t*));
  Subgraph *h_g_pool; cudaMalloc((void**)&h_g_pool, n*sizeof(Subgraph));
  Subgraph **h_gs; cudaMalloc((void**)&h_gs, n*sizeof(Subgraph*));
  int64_t *h_degacc, *h_deg2acc;
  cudaMalloc((void**)&h_degacc, n*sizeof(int64_t));
  cudaMalloc((void**)&h_deg2acc, n*sizeof(int64_t));

  // generally cudaMemcpy runs in h2d mode
  cudaMemcpy(h_adj_pool, adj_pool, adj_size*sizeof(bin_t), cudaMemcpyHostToDevice);
  cudaMemcpy(h_ptr_pool, ptr_pool, ptr_size*sizeof(bin_t), cudaMemcpyHostToDevice);
  cudaMemcpy(h_g_pool, g_pool, n*sizeof(Subgraph), cudaMemcpyHostToDevice);
  cudaMemcpy(h_gs, gs, n*sizeof(Subgraph*), cudaMemcpyHostToDevice);
  cudaMemcpy(h_degacc, ptr_offset, n*sizeof(int64_t), cudaMemcpyHostToDevice);
  cudaMemcpy(h_deg2acc, adj_offset, n*sizeof(int64_t), cudaMemcpyHostToDevice);

  int64_t *h_ans;
  cudaMalloc((void**)&h_ans, u_running*sizeof(int64_t));
  printf("Host: last error is '%s'\n", cudaGetErrorString(cudaGetLastError()));

  prepare_gpu<<<u_running, 1>>>(h_adj_pool, h_ptr_pool, h_g_pool, h_gs, h_degacc, h_deg2acc);
  cudaDeviceSynchronize();

overhead_timer.report("overhead");
exec_timer.reset();
  pivoting_traverse<<<u_running, config.blockdim>>>(h_gs, k, h_ans, dfs_pool, h_dfs_offset);
  cudaDeviceSynchronize();
exec_timer.report("kernel");
  printf("Host: last error is '%s'\n", cudaGetErrorString(cudaGetLastError()));

  pass
  int64_t *ans = new int64_t[u_running];
  memset(ans, 0, sizeof(int64_t)*u_running);
  cudaMemcpy(ans, h_ans, u_running*sizeof(int64_t), cudaMemcpyDeviceToHost);
  printf("copied u_running = %d\n", u_running);
  int64_t total_ans = 0;
  for (int i = 0; i < u_running; ++i) {
    total_ans += ans[i];
    // if (ans[i] != 0) printf("ans[%d] at host = %lld\n", i, ans[i]);
  }
  return total_ans;
}
};

#include <map>
bool checksimple(Graph *g) {
  std::map< std::pair<int, int>, int > f;
  int n = g->v_cnt;
  for (int i = 0; i < n; ++i)
    for (int p = g->vertex[i]; p < g->vertex[i+1]; ++p) {
      int j = g->edge[p];
      if (i == j) {
        eprintf("Contain self loop %d\n", i);
        return false;
      }
      if (i < j) f[std::make_pair(i, j)] += 1;
      else f[std::make_pair(j, i)] += 1;
    }
  for (auto p : f) if (p.second != 2) {
    eprintf("Edge %d-%d shows %d times", p.first.first, p.first.second, p.second);
    return false;
  }
  eprintf("Simple graph check passed\n");
  return true;
}

// This construction preserves order within each neighbour list.
// If you want it sorted, sort it beforehand.
Graph* wrap(const std::vector< std::vector<int> > &H) {
  Graph *h = new Graph;
  int n = H.size();
  int64_t m = 0;
  for (int i = 0; i < n; ++i) m += H[i].size();
  eprintf("Wrapping graph of (%d, %lld)\n", n, m);
  h->v_cnt = n;
  h->e_cnt = m;
  h->vertex = new int64_t[n + 1];
  h->edge = new int[m];
  int running = 0;
  for (int u = 0; u < n; ++u) {
    h->vertex[u] = running;
    for (int v : H[u]) {
      h->edge[running++] = v;
      sassert(0 <= v && v < n);
    }
  }
  h->vertex[n] = running;
  sassert(running == m);
  // h->tri_cnt = h->triangle_counting();
  h->tri_cnt = 0;
  return h;
}

bool my_loadg(Graph *&g, char* path) {
  // WARNING: tri_cnt is not set
  FILE *f = fopen(path, "r");
  int n; int64_t m;
  int res = fscanf(f, "%d %lld", &n, &m);
  if (res != 2) return false;
  if (!(0 < n && 0 < m)) return false;
  std::vector< std::vector<int> > G(n, std::vector<int>());
  for (int i = 0; i < m; ++i) {
    int x, y;
    int res = fscanf(f, "%d %d", &x, &y);
    if (res != 2) return false;
    if (!(0 <= x && x < n && 0 <= y && y < n)) return false;
    G[x].push_back(y);
    G[y].push_back(x);
  }
  for (int i = 0; i < n; ++i) std::sort(G[i].begin(), G[i].end());
  g = wrap(G);
  return true;
}

int64_t get_tricnt(Graph *G) {
  int n = G->v_cnt;
  int64_t ans = 0;
  std::vector<int> in(n);
  for (int u = 0; u < n; ++u) {
    for (int i = G->vertex[u]; i < G->vertex[u + 1]; ++i) {
      int v = G->edge[i];
      in[v] = 1;
    }
    for (int i = G->vertex[u]; i < G->vertex[u + 1]; ++i) {
      int v = G->edge[i];
      for (int j = G->vertex[v]; j < G->vertex[v + 1]; ++j) {
        int w = G->edge[j];
        if (in[w]) ans++;
      }
    }
    for (int i = G->vertex[u]; i < G->vertex[u + 1]; ++i) {
      int v = G->edge[i];
      in[v] = 0;
    }
  }
  sassert(ans % 6 == 0);
  return ans / 6;
}


using std::vector;
std::vector<int> order_degdescending(Graph *g) {
  int n = g->v_cnt;
  vector<int64_t> deg(n, 0);
  for (int i = 0; i < n; ++i) deg[i] = g->vertex[i + 1] - g->vertex[i];
  vector<int> nodes(n);
  for (int i = 0; i < n; ++i) nodes[i] = i;
  std::sort(nodes.begin(), nodes.end(), [&](int x, int y) {
    return deg[x] > deg[y];
  });
  return nodes;
}
// another deg-ascending order
std::vector<int> order_kcore(Graph *g) {
  int n = g->v_cnt;
  vector<int64_t> deg(n, 0);
  for (int i = 0; i < n; ++i) deg[i] = g->vertex[i + 1] - g->vertex[i];

  vector<int> nodes;
  
  vector<vector<int> > buk;
  buk.resize(*max_element(deg.begin(), deg.end()) + 1);
  for (int i = 0; i < n; ++i) buk[deg[i]].push_back(i);

  int ptr = 0;
  for (;;) {
    while (ptr < (int)buk.size() && buk[ptr].size() == 0) ptr++;
    if (ptr == (int)buk.size()) break;
    int u = buk[ptr].back(); buk[ptr].pop_back();
    if (deg[u] == -1) continue;
    nodes.push_back(u); deg[u] = -1;
    for (int64_t i = g->vertex[u]; i < g->vertex[u+1]; ++i) {
      int v = g->edge[i];
      if (deg[v] == -1) continue;
      buk[--deg[v]].push_back(v);
      if (deg[v] < ptr) ptr = deg[v];
    }
  }
  return nodes;
}
// another deg-ascending order
std::vector<int> order_revkcore(Graph *g) {
  int n = g->v_cnt;
  vector<int64_t> deg(n, 0);
  for (int i = 0; i < n; ++i) deg[i] = g->vertex[i + 1] - g->vertex[i];

  vector<int> nodes;
  
  vector<vector<int> > buk;
  buk.resize(*max_element(deg.begin(), deg.end()) + 1);
  for (int i = 0; i < n; ++i) buk[deg[i]].push_back(i);

  int ptr = buk.size() - 1;
  for (;;) {
    while (ptr >= 0 && buk[ptr].size() == 0) ptr--;
    if (ptr == -1) break;
    int u = buk[ptr].back(); buk[ptr].pop_back();
    if (ptr != deg[u]) continue;
    nodes.push_back(u); deg[u] = -1;
    for (int64_t i = g->vertex[u]; i < g->vertex[u+1]; ++i) {
      int v = g->edge[i];
      if (deg[v] == -1) continue;
      buk[--deg[v]].push_back(v);
    }
  }
  reverse(nodes.begin(), nodes.end());
  return nodes;
}

Graph* reorder(Graph *g, const vector<int> &nodes) {
  using std::vector;
  int n = g->v_cnt;
  vector<int> id(n, -1);
  for (int i = 0; i < n; ++i) id[nodes[i]] = i;
  for (int i = 0; i < n; ++i) sassert(id[i] != -1);
  // lets implement straightforward
  vector< vector<int> > H(n, vector<int>());
  for (int u = 0; u < n; ++u) {
    for (int i = g->vertex[u]; i < g->vertex[u+1]; ++i) {
      int v = g->edge[i];
      sassert(0 <= u && u < n && 0 <= v && v < n);
      H[id[u]].push_back(id[v]);
    }
  }
  for (int u = 0; u < n; ++u) std::sort(H[u].begin(), H[u].end());
  return wrap(H);
}

int main(int argc, char** argv) {
  Graph *original_g;
  DataLoader D;
  if (argc != 3 && argc != 4) {
    printf("usage: %s graph_file clique_size [blockdim]\n", argv[0]);
    return 0;
  }
  // 重要！这里得到的 original_g 里的边是有序的
  std::string path = argv[1];
  bool ok;
  if (path.back() == 'y') ok = my_loadg(original_g, argv[1]); // xxx_my
  else if (path.back() == 'g') { // xxx.g
    // 也就是 edge[vertex[u] ~ vertex[u+1]-1] 是有序数组
    // 后面的 g 亦是如此
    ok = D.fast_load(original_g, argv[1]);
  } else {
    printf("Cannot identify graph format. Aborted.");
    return 0;
  }
  if (!ok) {
    printf("Load data failed.\n");
    return 0;
  }

  // Graph *g = reorder_degascending(original_g);
  auto order = order_kcore(original_g);
  // std::reverse(order.begin(), order.end());
  Graph *g = reorder(original_g, order);
  int64_t sum = 0;
  int64_t maxdeg = 0;
  for (int i = 0; i < g->v_cnt; ++i) {
    int64_t deg = g->vertex[i + 1] - g->vertex[i];
    sum += deg * blocks(deg) * WIDTH / 8;
    maxdeg = std::max(maxdeg, deg);
  }
  // eprintf("Running simple graph check...\n");
  // sassert(checksimple(g));
  eprintf("v, e, tri_cnt: %d, %lld, %lld\n", g->v_cnt, g->e_cnt, g->tri_cnt);
  eprintf("sum deg^2, max deg: %lld, %lld\n", sum, maxdeg);
  if (maxdeg > MAXD) {
    eprintf("maxdeg > MAXD");
    return false;
  }


  int k = atoi(argv[2]);
  if (argc >= 4) Pivoting::config.blockdim = atoi(argv[3]);
  printf("Running with blockdim = %d\n", Pivoting::config.blockdim);
  sassert(Pivoting::config.valid());
  fflush(stdout);
  fflush(stderr);

  if (k == 1) {
    printf("count %d\n", original_g->v_cnt);
  } else if (k == 2) {
    printf("count %lld\n", original_g->e_cnt);
  } else {
    // k_clique_counting(original_g, k);
    // if (k == 3) {
    //   printf("cpu bf gives %lld\n", get_tricnt(g));
    // }
    int64_t ans = Pivoting::k_clique_counting_pivoting(g, k);
    printf("pivoting gives %lld\n", ans);
  }

  return 0;
}
