#pragma once

#include <gpu/config.cuh>
#include <gpu/vertex_set.cuh>
#include <gpu/utils.cuh>

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

// experimental. doesn't seem to be faster than the binary search version
__device__ uint32_t do_intersection_cached(
    uint32_t* out, const uint32_t* a, const uint32_t* b, uint32_t na, uint32_t nb
) {
    __shared__ uint32_t block_out_offset[THREADS_PER_BLOCK];
    __shared__ uint32_t block_out_size[WARPS_PER_BLOCK];
    __shared__ uint32_t block_cached_set[THREADS_PER_BLOCK];
    __shared__ bool block_temp[THREADS_PER_BLOCK]; // shared storage for reduction

    int wid = threadIdx.x / warpSize; // warp id
    int lid = threadIdx.x % warpSize; // lane id
    uint32_t *out_offset = block_out_offset + wid * warpSize;
    uint32_t *cached_set = block_cached_set + wid * warpSize;
    bool *temp = block_temp + wid * warpSize;
    uint32_t &out_size = block_out_size[wid];

    const uint32_t* const b_end = b + nb;
    const int nr_blocks = (nb + warpSize - 1) / warpSize;

    if (lid == 0)
        out_size = 0;

    int block = 0, cur_cache_block = -1;
    for (int nr_done = 0; nr_done < na; nr_done += warpSize) {
        uint32_t u = 0xffffffff;
        bool found = false;
        bool has_work = (nr_done + lid) < na;
        if (has_work)
            u = a[nr_done + lid];
        else
            found = true; // fake
        
        for (; block < nr_blocks; ++block) {
            if (cur_cache_block != block) {
                cur_cache_block = block;

                const uint32_t *local_b = b + block * warpSize;
                if (local_b + lid < b_end)
                    cached_set[lid] = local_b[lid];
                else
                    cached_set[lid] = 0xffffffff;
                __threadfence_block();
            }

            // could be binary search
            #pragma unroll 
            for (int i = 0; i < warpSize; ++i) {
                if (!found && (u == cached_set[i])) {
                    found = true;
                }
            }

            // warp reduction (and)
            temp[lid] = found || (cached_set[warpSize - 1] > u);
            for (int d = warpSize / 2; d >= 1; d >>= 1) {
                __threadfence_block();
                if (lid < d)
                    temp[lid] &= temp[lid + d];
            }
            bool stop = temp[0];
            if (stop)
                break;
        }

        bool has_output = has_work && found;
        out_offset[lid] = has_output;
        __threadfence_block();

        #pragma unroll
        for (int s = 1; s < warpSize; s *= 2) {
            uint32_t v = lid >= s ? out_offset[lid - s] : 0;
            // __threadfence_block();
            out_offset[lid] += v;
            __threadfence_block();
        }
        
        if (has_output) {
            uint32_t offset = out_offset[lid] - 1;
            out[out_size + offset] = u;
        }

        if (lid == 0)
            out_size += out_offset[warpSize - 1];
    }

    __threadfence_block();
    return out_size;
}
