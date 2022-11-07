#pragma once

#include <cstdint>
#include <cstdio>

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

constexpr int THREADS_PER_BLOCK = 256;
constexpr int THREADS_PER_WARP = 32;
constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / THREADS_PER_WARP;

#define get_edge_index(v, l, r) do { \
    l = vertex[v]; \
    r = vertex[v + 1]; \
} while(0)

template <typename T> __device__ inline void swap(T &a, T &b) {
    T t{std::move(a)};
    a = std::move(b);
    b = std::move(t);
}

#define gpu_check(result) gpu_assert((result), __FILE__, __LINE__)
inline void gpu_assert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA assertion failure: %s %s %d\n",
                cudaGetErrorString(code), file, line);
        exit(code);
    }
}

class GPUSchedule {
  public:
    inline __device__ int get_total_prefix_num() const {
        return total_prefix_num;
    }
    inline __device__ int get_basic_prefix_num() const {
        return basic_prefix_num;
    }
    inline __device__ int get_father_prefix_id(int prefix_id) const {
        return father_prefix_id[prefix_id];
    }
    inline __device__ int get_loop_set_prefix_id(int loop) const {
        return loop_set_prefix_id[loop];
    }
    inline __device__ int get_size() const { return size; }
    inline __device__ int get_last(int i) const { return last[i]; }
    inline __device__ int get_next(int i) const { return next[i]; }
    inline __device__ int get_break_size(int i) const { return break_size[i]; }
    inline __device__ int get_in_exclusion_optimize_num() const {
        return in_exclusion_optimize_num;
    }
    inline __device__ int get_total_restrict_num() const {
        return total_restrict_num;
    }
    inline __device__ int get_restrict_last(int i) const {
        return restrict_last[i];
    }
    inline __device__ int get_restrict_next(int i) const {
        return restrict_next[i];
    }
    inline __device__ int get_restrict_index(int i) const {
        return restrict_index[i];
    }
    // inline __device__ int get_k_val() const { return k_val;} // see below
    // (the k_val's definition line) before using this function

    int *adj_mat;
    int *father_prefix_id;
    int *last;
    int *next;
    int *break_size;
    int *loop_set_prefix_id;
    int *restrict_last;
    int *restrict_next;
    int *restrict_index;
    bool *only_need_size;
    // int* in_exclusion_optimize_val;
    // GPUGroupDim0 in_exclusion_optimize_group;
    // int in_exclusion_optimize_val_size;
    int size;
    int total_prefix_num;
    int basic_prefix_num;
    int total_restrict_num;
    int in_exclusion_optimize_num;
    // int k_val;

    int in_exclusion_optimize_vertex_id_size;
    int *in_exclusion_optimize_vertex_id;
    bool *in_exclusion_optimize_vertex_flag;
    int *in_exclusion_optimize_vertex_coef;

    int in_exclusion_optimize_array_size;
    int *in_exclusion_optimize_coef;
    bool *in_exclusion_optimize_flag;
    int *in_exclusion_optimize_ans_pos;

    uint32_t ans_array_offset;
};

__device__ void intersection2(uint32_t *tmp, const uint32_t *lbases,
                              const uint32_t *rbases, uint32_t ln, uint32_t rn,
                              uint32_t *p_tmp_size);
__device__ uint32_t do_intersection(uint32_t *, const uint32_t *,
                                    const uint32_t *, uint32_t, uint32_t);

class GPUVertexSet;
__device__ int unordered_subtraction_size(const GPUVertexSet &set0,
                                          const GPUVertexSet &set1,
                                          int size_after_restrict);

class GPUVertexSet {
  public:
    __device__ GPUVertexSet() {
        size = 0;
        data = NULL;
    }
    __device__ int get_size() const { return size; }
    __device__ uint32_t get_data(int i) const { return data[i]; }
    __device__ void push_back(uint32_t val) { data[size++] = val; }
    __device__ void pop_back() { --size; }
    __device__ uint32_t get_last() const { return data[size - 1]; }
    __device__ void set_data_ptr(uint32_t *ptr) { data = ptr; }
    __device__ uint32_t *get_data_ptr() const { return data; }
    __device__ bool
    has_data(uint32_t val) const // 注意：这里不用二分，调用它的是较小的无序集合
    {
        for (int i = 0; i < size; ++i)
            if (data[i] == val)
                return true;
        return false;
    }
    __device__ void init() { size = 0; }
    __device__ void init(uint32_t input_size, uint32_t *input_data) {
        size = input_size;
        data = input_data; //之后如果把所有prefix放到shared memory，由于input
                           // data在global
                           // memory上（因为是原图的边集），所以改成memcpy
    }
    __device__ void copy_from(const GPUVertexSet &other) //考虑改为并行
    {
        // 这个版本可能会有bank conflict
        uint32_t input_size = other.get_size(),
                 *input_data = other.get_data_ptr();
        size = input_size;
        int lid = threadIdx.x % THREADS_PER_WARP; // lane id
        int size_per_thread =
            (input_size + THREADS_PER_WARP - 1) / THREADS_PER_WARP;
        int start = size_per_thread * lid;
        int end = min(start + size_per_thread, input_size);
        for (int i = start; i < end; ++i)
            data[i] = input_data[i];
        __threadfence_block();
    }
    __device__ void build_vertex_set(const GPUSchedule *schedule,
                                     const GPUVertexSet *vertex_set,
                                     uint32_t *input_data, uint32_t input_size,
                                     int prefix_id) {
        int father_id = schedule->get_father_prefix_id(prefix_id);
        if (father_id == -1) {
            if (threadIdx.x % THREADS_PER_WARP == 0)
                init(input_size, input_data);
            __threadfence_block();
        } else {
            bool only_need_size = schedule->only_need_size[prefix_id];
            if (only_need_size) {
                if (threadIdx.x % THREADS_PER_WARP == 0)
                    init(input_size, input_data);
                __threadfence_block();
                if (input_size > vertex_set[father_id].get_size())
                    this->size -= unordered_subtraction_size(
                        *this, vertex_set[father_id], -1);
                else
                    this->size = vertex_set[father_id].get_size() -
                                 unordered_subtraction_size(
                                     vertex_set[father_id], *this, -1);
            } else {
                intersection2(this->data, vertex_set[father_id].get_data_ptr(),
                              input_data, vertex_set[father_id].get_size(),
                              input_size, &this->size);
            }
        }
    }

    __device__ void intersection_with(const GPUVertexSet &other) {
        uint32_t ret = do_intersection(data, data, other.get_data_ptr(), size,
                                       other.get_size());
        if (threadIdx.x % THREADS_PER_WARP == 0)
            size = ret;
        __threadfence_block();
    }

  private:
    uint32_t size;
    uint32_t *data;
};

/**
 * search-based intersection
 *
 * returns the size of the intersection set
 */
__device__ uint32_t do_intersection(uint32_t *out, const uint32_t *a,
                                    const uint32_t *b, uint32_t na,
                                    uint32_t nb) {
    __shared__ uint32_t block_out_offset[THREADS_PER_BLOCK];
    __shared__ uint32_t block_out_size[WARPS_PER_BLOCK];

    int wid = threadIdx.x / THREADS_PER_WARP; // warp id
    int lid = threadIdx.x % THREADS_PER_WARP; // lane id
    uint32_t *out_offset = block_out_offset + wid * THREADS_PER_WARP;
    uint32_t &out_size = block_out_size[wid];

    if (lid == 0)
        out_size = 0;

    for (int num_done = 0; num_done < na; num_done += THREADS_PER_WARP) {
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
__device__ void intersection2(uint32_t *tmp, const uint32_t *lbases,
                              const uint32_t *rbases, uint32_t ln, uint32_t rn,
                              uint32_t *p_tmp_size) {
    // make sure ln <= rn
    if (ln > rn) {
        swap(ln, rn);
        swap(lbases, rbases);
    }

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
__device__ int unordered_subtraction_size(const GPUVertexSet &set0,
                                          const GPUVertexSet &set1,
                                          int size_after_restrict = -1) {
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
    while (done1 < size1) {
        if (lid + done1 < size1) {
            int l = 0, r = size0 - 1;
            uint32_t val = set1.get_data(lid + done1);
            //考虑之后换一下二分查找的写法，比如改为l <
            // r，然后把mid的判断从循环里去掉，放到循环外(即最后l==r的时候)
            while (l <= r) {
                int mid = (l + r) >> 1;
                if (unlikely(set0.get_data(mid) == val)) {
                    atomicSub(&ret, 1);
                    break;
                }
                if (set0.get_data(mid) < val)
                    l = mid + 1;
                else
                    r = mid - 1;
            }
            // binary search
        }
        done1 += THREADS_PER_WARP;
    }

    __threadfence_block();
    return ret;
}
