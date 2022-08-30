// k <= 6
#define THRUST_IGNORE_CUB_VERSION_CHECK
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <omp.h>

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>

#include "../include/dataloader.h"
#include "../include/graph.h"

// constexpr int BITS_PER_UNSIGNED_INT = 32;

// 单个 partition_num = floor(size / 32)，表示多少个 uint 表示一个二进制集合
// int* partition_num;
int64_t* start_ptr;

constexpr int THREADS_PER_BLOCK = 64;

/**
 * @note: 工作假定 k <= 6，此时使用 degree orientation 更快（？）
 *        这里 degree orientation 和 degeneracy orientation
 * 两种方法都写了，用于作比较
 * @todo:
 */
void degree_orientation_init(Graph* original_g, Graph*& g) {
  g = new Graph();

  int* degree;
  int n = original_g->v_cnt;
  long long m = original_g->e_cnt;
  g->v_cnt = n;
  g->e_cnt = m >> 1;
  g->edge = (int*)malloc(g->e_cnt * sizeof(int));
  g->vertex = (int64_t*)malloc((g->v_cnt + 1) * sizeof(int64_t));

  degree = (int*)malloc(n * sizeof(int));
  for (int u = 0; u < n; ++u) {
    degree[u] = original_g->vertex[u + 1] - original_g->vertex[u];
  }

  int64_t ptr = 0;
  int max_degree = 0;
  for (int u = 0; u < n; ++u) {
    g->vertex[u] = ptr;
    for (int64_t i = original_g->vertex[u]; i < original_g->vertex[u + 1];
         ++i) {
      int v = original_g->edge[i];
      if ((degree[u] < degree[v]) || (degree[u] == degree[v] && u < v)) {
        g->edge[ptr++] = v;
      }
    }
    if (max_degree < ptr - g->vertex[u]) {
      max_degree = ptr - g->vertex[u];
    }
  }
  g->vertex[n] = ptr;
  assert(ptr == g->e_cnt);
  // partition_num = (max_degree - 1) / 32 + 1;
  assert(max_degree < THREADS_PER_BLOCK);
  printf("Maximum degree after orientation: %d\n", max_degree);
}

void degeneracy_orientation_init(Graph* original_g, Graph*& g) {
  g = new Graph();

  // 这里用 vector<int> node_with_degree[x] 来装当前度数为 x
  // 的所有节点，做到了线性求 k-core 序
  int* degree;
  int* order;
  int* vector_ptr;
  bool* in_vector;
  int n = original_g->v_cnt;
  long long m = original_g->e_cnt;
  std::vector<std::vector<int>> node_with_degree(n, std::vector<int>());
  g->v_cnt = n;
  g->e_cnt = m >> 1;
  g->edge = (int*)malloc(g->e_cnt * sizeof(int));
  g->vertex = (int64_t*)malloc((g->v_cnt + 1) * sizeof(int64_t));

  degree = (int*)malloc(n * sizeof(int));
  order = (int*)malloc(n * sizeof(int));
  in_vector = (bool*)malloc(n * sizeof(bool));
  vector_ptr = (int*)malloc(n * sizeof(int));

  for (int u = 0; u < n; ++u) {
    in_vector[u] = true;
    degree[u] = original_g->vertex[u + 1] - original_g->vertex[u];
    node_with_degree[degree[u]].push_back(u);
  }

  for (int i = 0; i < n; ++i) {
    vector_ptr[i] = 0;
  }

  // 精巧的实现
  int order_ptr = 0;
  for (int current_min_degree = 0; current_min_degree < n;
       ++current_min_degree) {
    bool back = false;
    for (int& i = vector_ptr[current_min_degree];
         i < node_with_degree[current_min_degree].size(); ++i) {
      int u = node_with_degree[current_min_degree][i];
      if (in_vector[u]) {
        order[u] = order_ptr++;
        in_vector[u] = false;
        for (int64_t j = original_g->vertex[u]; j < original_g->vertex[u + 1];
             ++j) {
          int v = original_g->edge[j];
          if (in_vector[v]) {
            node_with_degree[--degree[v]].push_back(v);
            if (degree[v] == current_min_degree - 1) {
              back = true;
            }
          }
        }
      }
    }
    if (back) {
      // 由于 for 循环里还要 +1，这里实际上只减了 1
      current_min_degree -= 2;
    }
  }

  int64_t ptr = 0;
  int max_degree = 0;
  for (int u = 0; u < n; ++u) {
    g->vertex[u] = ptr;
    for (int64_t i = original_g->vertex[u]; i < original_g->vertex[u + 1];
         ++i) {
      int v = original_g->edge[i];
      if (order[u] < order[v]) {
        g->edge[ptr++] = v;
      }
    }
    if (max_degree < ptr - g->vertex[u]) {
      max_degree = ptr - g->vertex[u];
    }
  }
  g->vertex[n] = ptr;
  assert(ptr == g->e_cnt);
  // partition_num = (max_degree - 1) / 32 + 1;
  assert(max_degree < THREADS_PER_BLOCK);
  printf("Maximum vertex degree after orientation: %d\n", max_degree);
}


/*
__device__ int popcount(unsigned int* x, int* partition_num) {
  int result = 0;
  for (int i = 0; i < *partition_num; ++i) {
    result += __popc(x[i]);
  }
  return result;
}
*/

constexpr int MAX_THREAD_BLOCK = 0x0FFFFFFF;  // 好像可以设成 2^31-1？
constexpr int MAX_DEPTH = 8;
constexpr int THREADS_PER_WARP = 32;
constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / THREADS_PER_WARP;

__device__ unsigned long long dev_sum = 0;

/**
 * @note: 每一个 Warp 负责一棵搜索子树，Warp 里面的线程一起负责求交
 *        后面的版本是最开始写的暴力：线程块内的每一个线程独立处理一棵搜索子树
 * @todo: 1. Warp 细化（有时候 32 不是最优）
 *        2. 感觉可以加启发式（类似木桶理论），而不是严格按照 vertex 编号顺序来
 */
__global__ void traverse_on_warp_partition(int* n, int* thread_block_num,
                                           unsigned int* binary_adj,
                                           int64_t* vertex, int64_t* start_ptr,
                                           int* k) {
  __shared__ unsigned long long cache[WARPS_PER_BLOCK];
  // __shared__ int stack_vertex_pool[WARPS_PER_BLOCK * MAX_DEPTH];
  // __shared__ int* stack_vertex_ptr[WARPS_PER_BLOCK];
  __shared__ int stack_binary_adj_pool[THREADS_PER_BLOCK * MAX_DEPTH];
  __shared__ int* stack_binary_adj_ptr[WARPS_PER_BLOCK];
  int stack_vertex[MAX_DEPTH];
  // int stack_binary_adj[MAX_DEPTH * THREADS_PER_WARP];

  if (threadIdx.x == 0) {
    for (int i = 0; i < WARPS_PER_BLOCK; ++i) {
      cache[i] = 0;
    }
    for (int i = 0; i < WARPS_PER_BLOCK; ++i) {
      // stack_vertex_ptr[i] = stack_vertex_pool + i * MAX_DEPTH;
      stack_binary_adj_ptr[i] =
          stack_binary_adj_pool + i * THREADS_PER_WARP * MAX_DEPTH;
    }
  }
  __syncthreads();

  // int u = blockIdx.x;
  int warpidx = threadIdx.x / THREADS_PER_WARP;
  // int size = (int)(vertex[u + 1] - vertex[u]);
  bool is_main_thread = (threadIdx.x % THREADS_PER_WARP == 0);
  // int* stack_vertex = stack_vertex_ptr[warpidx];
  int* stack_binary_adj = stack_binary_adj_ptr[warpidx];

  for (int64_t o = (int64_t)blockIdx.x * WARPS_PER_BLOCK + warpidx;
       o < vertex[*n]; o += (int64_t)thread_block_num * WARPS_PER_BLOCK) {
    int l = 0, r = *n - 1;
    while (l != r) {
      int middle = (l + r >> 1) + 1;
      if (vertex[middle] <= o) {
        l = middle;
      } else {
        r = middle - 1;
      }
    }
    // u 是结点编号，v 实际上是在 adj(u) 中的相对编号
    int u = l;
    int v = (int)(o - vertex[u]);
    int size = (int)(vertex[u + 1] - vertex[u]);
    // ceil(size / 32)
    int partition_num = (size - 1) / 32 + 1;

    int top = 0;
    stack_vertex[top] = v;
    stack_vertex[top + 1] = -1;
    if (is_main_thread) {
      for (int i = 0; i < partition_num; ++i) {
        stack_binary_adj[i] = binary_adj[start_ptr[u] + v * partition_num + i];
      }
    }
    // __syncwarp();

    unsigned long long sum = 0;
    // 预定义 实际上是用于迭代的。好像会快一些？
    int partitionidx;
    int bitidx;
    bool ok;
    int foobar;

    // 因为 partition_num 的定义后面改过，所以下面注释里的一些地方可能有问题
    while (~top) {
      if (top + 3 == *k) {
        if (is_main_thread) {
          // sum += popcount(stack_binary_adj + (*partition_num) * top,
          // partition_num);
          foobar = partition_num * top;
          for (int i = 0; i < partition_num; ++i) {
            sum += __popc(stack_binary_adj[foobar + i]);
          }
        }
        --top;
      } else {
        ++stack_vertex[top + 1];

        partitionidx = stack_vertex[top + 1] >> 5;  // / 32
        bitidx = stack_vertex[top + 1] & 31;        // % 32
        ok = false;

        while (stack_vertex[top + 1] < size) {
          // 小优化：如果后面都没有 1 位了，直接跳过（最多）32 个点
          // 不过实际效果好像并不是很显著？
          // if (!(stack_binary_adj[top * (*partition_num) + partitionidx] >>
          // bitidx)) {
          if (!(binary_adj[start_ptr[u] + stack_vertex[top] * partition_num +
                           partitionidx] >>
                bitidx)) {
            stack_vertex[top + 1] += 32 - bitidx;
            bitidx = 0;
            ++partitionidx;
            continue;
          }
          // if ((stack_binary_adj[top * (*partition_num) + partitionidx] >>
          // bitidx) & 1) {
          if ((binary_adj[start_ptr[u] + stack_vertex[top] * partition_num +
                          partitionidx] >>
               bitidx) &
              1) {
            ++top;
            int i = threadIdx.x % THREADS_PER_WARP;
            if (i < partition_num) {
              stack_binary_adj[top * partition_num + i] =
                  stack_binary_adj[(top - 1) * partition_num + i] &
                  binary_adj[start_ptr[u] + stack_vertex[top] * partition_num +
                             i];
            }
            // __syncwarp();

            ok = true;
            stack_vertex[top + 1] = -1;
            break;
          }
          ++stack_vertex[top + 1];
          if (++bitidx == 32) {
            bitidx = 0;
            ++partitionidx;
          }
        }

        if (!ok) {
          --top;
        }
      }
    }

    if (is_main_thread) {
      cache[warpidx] += sum;
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    for (int i = 1; i < WARPS_PER_BLOCK; ++i) {
      cache[0] += cache[i];
    }
    atomicAdd(&dev_sum, cache[0]);
  }
}

__global__ void traverse(unsigned int* binary_adj, int64_t* vertex,
                         int64_t* start_ptr, int* k) {

  typedef cub::BlockReduce<unsigned long long, THREADS_PER_BLOCK> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  int u = blockIdx.x;
  int tid = threadIdx.x;
  int size = (int)(vertex[u + 1] - vertex[u]);
  int partition_num = (size - 1) / 32 + 1;

  // int* stack_vertex = new int[MAX_DEPTH];
  // unsigned int* stack_binary_adj = new unsigned int[MAX_DEPTH * 18];
  int stack_vertex[MAX_DEPTH];
  unsigned int stack_binary_adj[MAX_DEPTH * 32];

  assert(partition_num < 32);

  unsigned long long sum = 0;


  for(int v = tid; v < size; v += THREADS_PER_BLOCK) {
    int top = 0;

    stack_vertex[top] = v;
    stack_vertex[top + 1] = -1;
    for (int i = 0; i < partition_num; ++i) {
      stack_binary_adj[i] = binary_adj[start_ptr[u] + v * partition_num + i];
    }


    int partitionidx;
    int bitidx;
    bool ok;
    int foobar;

    while (~top) {
      if (top + 3 == *k) {
        // sum += popcount(stack_binary_adj + (*partition_num) * top,
        // partition_num);
        foobar = partition_num * top;
        for (int i = 0; i < partition_num; ++i) {
          sum += __popc(stack_binary_adj[foobar + i]);
        }
        --top;
      } else {
        ++stack_vertex[top + 1];

        partitionidx = stack_vertex[top + 1] >> 5;
        bitidx = stack_vertex[top + 1] & 31;
        ok = false;

        while (stack_vertex[top + 1] < size) {
          if (!(stack_binary_adj[top * partition_num + partitionidx] >>
                bitidx)) {
            stack_vertex[top + 1] += 32 - bitidx;
            bitidx = 0;
            ++partitionidx;
            continue;
          }
          if ((stack_binary_adj[top * partition_num + partitionidx] >> bitidx) &
              1) {
            ++top;
            for (int i = 0; i < partition_num; ++i) {
              stack_binary_adj[top * partition_num + i] =
                  stack_binary_adj[(top - 1) * partition_num + i] &
                  binary_adj[start_ptr[u] + stack_vertex[top] * partition_num +
                             i];
            }
            ok = true;
            stack_vertex[top + 1] = -1;
            break;
          }
          ++stack_vertex[top + 1];
          if (++bitidx == 32) {
            bitidx = 0;
            ++partitionidx;
          }
        }

        if (!ok) {
          --top;
        }
      }
    }


  }

  if(sum > 0) {
    assert(false);
  }

  __syncthreads();
  
  unsigned long long aggregate = BlockReduce(temp_storage).Sum(sum);

  if(tid == 0) {
    atomicAdd(&dev_sum, aggregate);
  }

  // delete[] stack_vertex;
  // delete[] stack_binary_adj;
}

void k_clique_counting(Graph* g, int k) {
  int n = g->v_cnt;
  long long m = g->e_cnt;
  start_ptr = new int64_t[n + 1];

  start_ptr[0] = 0;
  for (int u = 0; u < n; ++u) {
    int size = g->vertex[u + 1] - g->vertex[u];
    start_ptr[u + 1] = start_ptr[u] + size * ((size - 1) / 32 + 1);
  }

  unsigned int* binary_adj = new unsigned int [start_ptr[n]];
  for (long long i = start_ptr[n + 1] - 1; ~i; --i) {
    binary_adj[i] = 0;
  }

  #pragma omp parallel for num_threads(16)
  for (int u = 0; u < n; ++u) {
    for (int64_t i = g->vertex[u]; i < g->vertex[u + 1]; ++i) {
      int v = g->edge[i];
      int size = g->vertex[u + 1] - g->vertex[u];
      int partition_num = (size - 1) / 32 + 1;

      int64_t p = g->vertex[u];
      int64_t q = g->vertex[v];
      int partitionidx = 0;
      int bitidx = 0;
      while (p < g->vertex[u + 1]) {
        while (q < g->vertex[v + 1] && g->edge[q] < g->edge[p]) {
          ++q;
        }
        if (q < g->vertex[v + 1] && g->edge[p] == g->edge[q]) {
          binary_adj[start_ptr[u] + partition_num * (i - g->vertex[u]) +
                     partitionidx] |= 1 << bitidx;
        }
        ++p;
        if (++bitidx == 32) {
          bitidx = 0;
          ++partitionidx;
        }
      }
    }
  }

  // printf("%d %lld\n", g->v_cnt, g->e_cnt);
  // for (int i = 0; i <= g->v_cnt; ++i) {
  //   std::cerr << g->vertex[i] << " \n"[i == g->v_cnt];
  // }
  // for (long long i = 0; i < g->e_cnt; ++i) {
  //   std::cerr << binary_adj[i] << " \n"[i + 1 == g->e_cnt];
  // }

  unsigned int* gpu_binary_adj;
  int64_t* gpu_vertex;
  int64_t* gpu_start_ptr;
  int* gpu_k;
  int* gpu_n;
  int* gpu_thread_block_num;
  int thread_block_num =
      (int)std::min((long long)MAX_THREAD_BLOCK,
                    (m * THREADS_PER_WARP - 1) / THREADS_PER_BLOCK + 1);

  cudaMalloc((void**)&gpu_binary_adj, start_ptr[n] * sizeof(unsigned int));
  cudaMalloc((void**)&gpu_vertex, (n + 1) * sizeof(int64_t));
  cudaMalloc((void**)&gpu_start_ptr, (n + 1) * sizeof(int64_t));
  cudaMalloc((void**)&gpu_k, sizeof(int));
  cudaMalloc((void**)&gpu_n, sizeof(int));
  cudaMalloc((void**)&gpu_thread_block_num, sizeof(int));

  cudaMemcpy(gpu_binary_adj, binary_adj, start_ptr[n] * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_vertex, g->vertex, (n + 1) * sizeof(int64_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_start_ptr, start_ptr, (n + 1) * sizeof(int64_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_k, &k, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_n, &n, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_thread_block_num, &thread_block_num, sizeof(int),
             cudaMemcpyHostToDevice);

  clock_t start_time = clock();
  printf("Counting function is running...\n");

  unsigned long long sum = 0;
  cudaMemcpyToSymbol(dev_sum, &sum, sizeof(unsigned long long));

  traverse<<<n, THREADS_PER_BLOCK>>>(gpu_binary_adj, gpu_vertex, gpu_start_ptr,
                                     gpu_k);
  // traverse_on_warp_partition<<<thread_block_num, THREADS_PER_BLOCK>>>
  //   (gpu_n, gpu_thread_block_num, gpu_binary_adj, gpu_vertex, gpu_start_ptr,
  //   gpu_k);
  cudaDeviceSynchronize();

  cudaMemcpyFromSymbol(&sum, dev_sum, sizeof(unsigned long long));
  printf("count %llu\n", sum);

  clock_t end_time = clock();
  printf("Time: %.3lfms.\n",
         (double)(end_time - start_time) / CLOCKS_PER_SEC * 1000.0);

  cudaFree(gpu_binary_adj);
  cudaFree(gpu_vertex);
  cudaFree(gpu_start_ptr);
  cudaFree(gpu_k);
  cudaFree(gpu_n);
  cudaFree(gpu_thread_block_num);
}

// test
/*
int main() {
  Graph *original_g = new Graph();
  Graph *g;
  int k;

  freopen("data/3.in", "r", stdin);
  scanf("%d%lld%d", &(original_g->v_cnt), &(original_g->e_cnt), &k);
  original_g->edge = (int*)malloc(original_g->e_cnt * sizeof(int));
  original_g->vertex = (int64_t*)malloc((original_g->v_cnt + 1) *
sizeof(int64_t)); for (long long i = 0; i < original_g->e_cnt; ++i) {
    scanf("%d", &original_g->edge[i]);
  }
  for (int i = 0; i <= original_g->v_cnt; ++i) {
    scanf("%lld", &original_g->vertex[i]);
  }

  orientation_init(original_g, g);
  // std::cerr << partition_num << '\n';

  // printf("%d %lld\n", g->v_cnt, g->e_cnt);
  // for (int i = 0; i <= g->v_cnt; ++i) {
  //   std::cerr << g->vertex[i] << " \n"[i == g->v_cnt];
  // }
  // for (long long i = 0; i < g->e_cnt; ++i) {
  //   std::cerr << g->edge[i] << " \n"[i + 1 == g->e_cnt];
  // }

  clock_t start_time = clock();

  if (k == 1) {
    printf("count %d\n", g->v_cnt);
  } else if (k == 2) {
    printf("count %lld\n", g->e_cnt);
  } else {
    k_clique_counting(g, k);
    cudaDeviceSynchronize();
  }

  clock_t end_time = clock();
  printf("Time: %lfs.\n", (double)((end_time - start_time) / CLOCKS_PER_SEC));
  return 0;
}
*/

int main(int argc, char** argv) {
  Graph *original_g, *g;
  DataLoader D;

  if (argc != 3) {
    printf("usage: %s graph_file clique_size\n", argv[0]);
    return 0;
  }
  // 重要！这里得到的 original_g 里的边是有序的
  // 也就是 edge[vertex[u] ~ vertex[u+1]-1] 是有序数组
  // 后面的 g 亦是如此
  bool ok = D.fast_load(original_g, argv[1]);
  if (!ok) {
    printf("Load data failed.\n");
    return 0;
  }

  degree_orientation_init(original_g, g);
  // degeneracy_orientation_init(original_g, g);

  int k = atoi(argv[2]);
  if (k == 1) {
    printf("count %d\n", g->v_cnt);
  } else if (k == 2) {
    printf("count %lld\n", g->e_cnt);
  } else {
    k_clique_counting(g, k);
  }

  return 0;
}