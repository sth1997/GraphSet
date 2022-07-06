#pragma once

#include <cstdint>

template <typename T>
__device__ __host__ int search(const T a[], T t, int n)
{
    int m, l = 0, r = n;
    while (l < r) {
        m = (l + r) / 2;
        if (a[m] < t)
            l = m + 1;
        else
            r = m;
    }
    return r;
}

template <typename T>
__device__ __host__ bool search2(const T a[], T t, int n)
{
    int mid, l = 0, r = n - 1;
    while (l <= r) {
        mid = (l + r) / 2;
        if (a[mid] < t) {
            l = mid + 1;
        } else if (a[mid] > t) {
            r = mid - 1;
        } else {
            return 1;
        }
    }
    return 0;
}

template <typename T>
__device__ __host__ int merge(T t[], const T a[], const T b[], int n, int m)
{
    int i, j, k;
    i = j = k = 0;
    while (i < n && j < m) {
        T x = a[i], y = b[j];
        if (x < y)
            ++i;
        else if (x > y)
            ++j;
        else {
            t[k++] = x;
            ++i, ++j;
        }
    }
    return k;
}

template <typename T>
__device__ void warp_inclusive_scan(T out_offset[])
{
    int lid = threadIdx.x % warpSize;
    #pragma unroll
    for (int s = 1; s < warpSize; s <<= 1) {
        T v = (lid >= s) ? out_offset[lid - s] : 0;
        out_offset[lid] += v;
        __threadfence_block();
    }
}

__device__ uint32_t warp_binary_search_intersection(uint32_t* out_offset, uint32_t& out_size,
    uint32_t* c, const uint32_t* a, const uint32_t* b, int n, int m)
{
    int lid = threadIdx.x % warpSize;
    if (lid == 0)
        out_size = 0;
    
    for (int i = 0; i < n; i += warpSize) {
        bool found = 0;
        uint32_t u = 0;
        if (i + lid < n) {
            u = a[i + lid];
            found = search2(b, u, m);
        }
        out_offset[lid] = found;
        __threadfence_block();

        warp_inclusive_scan(out_offset);

        if (found) {
            int offset = out_offset[lid] - 1;
            c[out_size + offset] = u;
        }

        if (lid == 0)
            out_size += out_offset[warpSize - 1];
    }
    return out_size;
}

__device__ uint32_t warp_parallel_merge_intersection(uint32_t* out_offset, uint32_t* border,
    uint32_t* c, uint32_t* t, const uint32_t* a, const uint32_t* b, int n, int m)
{
    int lid = threadIdx.x % warpSize;
    int a_blk_sz = (n + warpSize - 1) / warpSize;

    // partition
    int a_begin = lid * a_blk_sz;
    if (a_begin < n)
        border[lid] = lid ? search(b, a[a_begin], m) : 0;
    else {
        a_begin = n;
        border[lid] = m;
    }
    __threadfence_block();

    // linear merge
    int a_end = min(a_begin + a_blk_sz, n);
    int b_begin = border[lid];
    int b_end = (lid != warpSize - 1) ? border[lid + 1] : m;
    int local_cnt = merge(&t[a_begin], &a[a_begin], &b[b_begin], a_end - a_begin, b_end - b_begin);
    // printf("thread %d: a [%d, %d) b [%d, %d) cnt = %d\n", lid, a_begin, a_end, b_begin, b_end, local_cnt);

    // compact
    out_offset[lid] = local_cnt;
    __threadfence_block();

    warp_inclusive_scan(out_offset);
    auto from = &t[a_begin];
    auto to = &c[out_offset[lid] - local_cnt];
    // printf("thread %d: dst range [%d, %d)\n", lid, out_offset[lid] - local_cnt, out_offset[lid]);
    for (int i = 0; i < local_cnt; ++i)
        to[i] = from[i];
    return out_offset[warpSize - 1];
}
