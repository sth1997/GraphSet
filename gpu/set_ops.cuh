/**
 * @warning: 请注意这里所有涉及集合大小相关的变量全部使用了int，而非uint32_t
 * 目前不支持集合操作中存在大小>= 2^31的集合（通常也到不了这么大）
 * 
 * 比较标准的做法是使用size_t。但全部换为相关类型后似乎寄存器使用数量上升了，并且发生了local memory spill。
 * 暂时维持现状。
 */
#include "common.cuh"

/**
 * @brief search-based intersection
 * 
 * @return the size of the intersection set
 * 
 * @note：A和B并不是地位相等的。如果要进行in-place操作，请把输入放在A而不是B。
 * @todo：shared memory缓存优化
 */
__device__ int do_intersection(uint32_t* out, const uint32_t* A, const uint32_t* B, int A_size, int B_size)
{
    __shared__ int block_out_offset[THREADS_PER_BLOCK];
    __shared__ int block_out_size[WARPS_PER_BLOCK];

    int wid = threadIdx.x / THREADS_PER_WARP; // warp id
    int lid = threadIdx.x % THREADS_PER_WARP; // lane id
    auto *out_offset = block_out_offset + wid * THREADS_PER_WARP;
    auto &out_size = block_out_size[wid];

    if (lid == 0)
        out_size = 0;

    for (int num_done = 0; num_done < A_size; num_done += THREADS_PER_WARP) {
        bool found = 0;
        uint32_t a = 0;
        if (num_done + lid < A_size) {
            a = A[num_done + lid]; // u: an element in set a
            int mid, l = 0, r = int(B_size) - 1;
            while (l <= r) {
                mid = (l + r) >> 1;
                if (B[mid] < a) {
                    l = mid + 1;
                } else if (B[mid] > a) {
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
            // __threadfence_block(); // 在以warp为基本单位执行时，这句应该不需要吧？
            out_offset[lid] += v;
            __threadfence_block();
        }
        
        if (found) {
            auto offset = out_offset[lid] - 1;
            out[out_size + offset] = a;
        }

        if (lid == 0)
            out_size += out_offset[THREADS_PER_WARP - 1];
    }

    __threadfence_block();
    return out_size;
}

__device__ int do_intersection(uint32_t* out, const uint32_t* A, int A_size, const GPUVertexSet& B)
{
    return do_intersection(out, A, B.get_data_ptr(), A_size, B.get_size());
}

__device__ int do_intersection(uint32_t* out, const GPUVertexSet& A, const GPUVertexSet& B)
{
    return do_intersection(out, A.get_data_ptr(), B.get_data_ptr(), A.get_size(), B.get_size());
}

__device__ int set_intersection_size(const uint32_t* A, const uint32_t* B, int A_size, int B_size)
{
    __shared__ int block_ret[WARPS_PER_BLOCK];

    int wid = threadIdx.x / THREADS_PER_WARP;
    int lid = threadIdx.x % THREADS_PER_WARP;
    auto &ret = block_ret[wid];
    if (lid == 0)
        ret = 0;

    for (int i = lid; i < A_size; i += THREADS_PER_WARP) {
        int mid, l = 0, r = int(B_size) - 1;
        uint32_t a = A[i];
        while (l <= r) {
            mid = (l + r) >> 1;
            uint32_t b = B[mid];
            if (unlikely(b == a)) {
                atomicAdd(&ret, 1);
                break;
            }
            if (b < a)
                l = mid + 1;
            else
                r = mid - 1;
        }
    }
    
    __threadfence_block();
    return ret;
}

/**
 * @brief calculate |A - D|
 * @note A should be an ordered set
 */
__device__ int set_difference_size(const uint32_t* A, const uint32_t* D, int A_size, int D_size)
{
    __shared__ int block_ret[WARPS_PER_BLOCK];

    int wid = threadIdx.x / THREADS_PER_WARP;
    int lid = threadIdx.x % THREADS_PER_WARP;
    auto &ret = block_ret[wid];
    if (lid == 0)
        ret = A_size;
    // __threadfence_block(); // 也不用吧？有atomicSub和最后的barrier顶着

    for (int i = lid; i < D_size; i += THREADS_PER_WARP) {
        uint32_t d = D[i];
        int mid, l = 0, r = int(A_size) - 1;
        //考虑之后换一下二分查找的写法，比如改为l < r，然后把mid的判断从循环里去掉，放到循环外(即最后l==r的时候)
        while (l <= r) {
            mid = (l + r) >> 1;
            uint32_t a = A[mid];
            if (unlikely(a == d)) {
                atomicSub(&ret, 1);
                break;
            }
            if (a < d)
                l = mid + 1;
            else
                r = mid - 1;
        }
    }
    
    __threadfence_block();
    return ret;
}

__device__ int set_difference_size(const uint32_t* A, int A_size, const GPUSubtractionSet& D)
{
    return set_difference_size(A, D.get_data_ptr(), A_size, D.get_size());
}

__device__ int set_difference_size(const GPUVertexSet& A, const GPUSubtractionSet& D)
{
    return set_difference_size(A.get_data_ptr(), D.get_data_ptr(), A.get_size(), D.get_size());
}

constexpr int MAX_SUBTRACTION_COUNT = 4;
/**
 * @brief caculate |A_i - D| for each A_i
 */
__device__ int* set_difference_size_compacted(const uint32_t* As[], const uint32_t* D, const int A_sizes[], int D_size, int nr_sets)
{
    __shared__ int block_ret[MAX_SUBTRACTION_COUNT * WARPS_PER_BLOCK];

    int wid = threadIdx.x / THREADS_PER_WARP;
    int lid = threadIdx.x % THREADS_PER_WARP;
    
    auto total_D_size = D_size * nr_sets;
    auto *ret = block_ret + wid * MAX_SUBTRACTION_COUNT;
    if (lid < nr_sets)
        ret[lid] = A_sizes[lid];

    for (int i = lid; i < total_D_size; i += THREADS_PER_WARP) {
        int set_idx = i / D_size;
        uint32_t d = D[i % D_size];

        const uint32_t *A = As[set_idx];
        int mid, l = 0, r = int(A_sizes[set_idx]) - 1;
        while (l <= r) {
            mid = (l + r) >> 1;
            uint32_t a = A[mid];
            if (unlikely(a == d)) {
                atomicSub(&ret[set_idx], 1);
                break;
            }
            if (a < d)
                l = mid + 1;
            else
                r = mid - 1;
        }
    }

    __threadfence_block();
    return ret;
}

__device__ int* set_difference_size_compacted(const uint32_t* As[], const int A_sizes[], const GPUSubtractionSet& D, int nr_sets)
{
    return set_difference_size_compacted(As, D.get_data_ptr(), A_sizes, D.get_size(), nr_sets);
}

/**
 * @brief caculate |A_0 - D|, |A_1 - D|
 */
__device__ int* set_difference_size_c2(const uint32_t* A0, int A0_size, const uint32_t* A1, int A1_size, const GPUSubtractionSet& D)
{
    __shared__ int block_ret[2 * WARPS_PER_BLOCK];

    int wid = threadIdx.x / THREADS_PER_WARP;
    int lid = threadIdx.x % THREADS_PER_WARP;
    
    int D_size = D.get_size();
    auto total_D_size = D_size * 2;
    auto *ret = block_ret + wid * 2;
    if (lid == 0) {
        ret[0] = A0_size;
        ret[1] = A1_size;
    }

    for (int i = lid; i < total_D_size; i += THREADS_PER_WARP) {
        int set_idx = i / D_size;
        uint32_t d = D.get_data(i % D_size);

        const uint32_t *A = set_idx ? A1 : A0;
        int mid, l = 0, r = int(set_idx ? A1_size : A0_size) - 1;
        while (l <= r) {
            mid = (l + r) >> 1;
            uint32_t a = A[mid];
            if (unlikely(a == d)) {
                atomicSub(&ret[set_idx], 1);
                break;
            }
            if (a < d)
                l = mid + 1;
            else
                r = mid - 1;
        }
    }

    __threadfence_block();
    return ret;
}

__device__ int* set_difference_size_c2(const uint32_t* A0, int A0_size, const GPUVertexSet& A1, const GPUSubtractionSet& D)
{
    return set_difference_size_c2(A0, A0_size, A1.get_data_ptr(), A1.get_size(), D);
}

__device__ int* set_difference_size_c2(const GPUVertexSet& A0, const GPUVertexSet& A1, const GPUSubtractionSet& D)
{
    return set_difference_size_c2(A0.get_data_ptr(), A0.get_size(), A1.get_data_ptr(), A1.get_size(), D);
}

/**
 * @brief calculate |A ∩ B - D|
 */
__device__ int set_fused_op1(const uint32_t* A, int A_size, const GPUVertexSet& B, const GPUSubtractionSet& D)
{
    __shared__ int block_ret[WARPS_PER_BLOCK];

    int wid = threadIdx.x / THREADS_PER_WARP;
    int lid = threadIdx.x % THREADS_PER_WARP;

    auto &ret = block_ret[wid];
    if (lid == 0)
        ret = 0;
    
    for (int i = lid; i < A_size; i += THREADS_PER_WARP) {
        uint32_t a = A[i];
        bool found = 0;
        int mid, l = 0, r = int(B.get_size()) - 1;
        while (l <= r) {
            mid = (l + r) >> 1;
            uint32_t b = B.get_data(mid);
            if (unlikely(b == a)) {
                found = 1;
                break;
            }
            if (b < a)
                l = mid + 1;
            else
                r = mid - 1;
        }

        if (found && !D.has_data_unrolled(a))
            atomicAdd(&ret, 1);
    }

    __threadfence_block();
    return ret;
}

__device__ int set_fused_op1(const GPUVertexSet& A, const GPUVertexSet& B, const GPUSubtractionSet& D)
{
    return set_fused_op1(A.get_data_ptr(), A.get_size(), B, D);
}

/**
 * @brief calculate |A_0 ∩ B - D|, |A_1 ∩ B - D|
 */
__device__ int* set_fused_op2(const GPUVertexSet& A0, const GPUVertexSet& A1, const GPUVertexSet& B, const GPUSubtractionSet& D)
{
    __shared__ int block_ret[2 * WARPS_PER_BLOCK];

    int wid = threadIdx.x / THREADS_PER_WARP;
    int lid = threadIdx.x % THREADS_PER_WARP;

    auto *ret = block_ret + wid * 2;
    if (lid == 0)
        ret[0] = ret[1] = 0;
    
    int A0_size = A0.get_size();
    int A1_size = A1.get_size();
    auto total_A_size = A0_size + A1_size;
    for (int i = lid; i < total_A_size; i += THREADS_PER_WARP) {
        uint32_t a;
        int set_idx;
        if (i < A0_size)
            a = A0.get_data(i), set_idx = 0;
        else
            a = A1.get_data(i - A0_size), set_idx = 1;
        
        bool found = 0;
        int mid, l = 0, r = int(B.get_size()) - 1;
        while (l <= r) {
            mid = (l + r) >> 1;
            uint32_t b = B.get_data(mid);
            if (unlikely(b == a)) {
                found = 1;
                break;
            }
            if (b < a)
                l = mid + 1;
            else
                r = mid - 1;
        }

        if (found && !D.has_data_unrolled(a))
            atomicAdd(&ret[set_idx], 1);
    }

    __threadfence_block();
    return ret;
}
