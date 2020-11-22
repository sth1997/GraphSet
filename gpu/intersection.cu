#include <algorithm>
#include <utility>
#include <cstdlib>
#include <cstdio>

constexpr int THREADS_PER_BLOCK = 64;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

template <typename T>
__device__ inline void swap(T& a, T& b)
{
    T t(std::move(a));
    a = std::move(b);
    b = std::move(t);
}

__global__ void intersection(uint32_t *tmp, uint32_t *lbases, uint32_t *rbases, uint32_t ln, uint32_t rn, uint32_t* p_tmp_size)
{
    __shared__ uint32_t tmp_offset[THREADS_PER_BLOCK];
    __shared__ uint32_t tmp_size;

    // make sure ln <= rn
    if (ln > rn) {
        swap(ln, rn);
        swap(lbases, rbases);
    }

    bool verbose = 0;
    if (threadIdx.x == 0) {
        tmp_size = 0;
        printf("block %d ln:%d rn:%d\n", blockIdx.x, ln, rn);
        printf("L: ");
        for (int i = 0; i < ln; ++i)
            printf("%d ", lbases[i]);
        printf("\nR: ");
        for (int i = 0; i < rn; ++i)
            printf("%d ", rbases[i]);
        printf("\n");
    }

    uint32_t lsize, i = 0;
    while (i < ln) {
        lsize = min(ln - i, THREADS_PER_BLOCK);

        //if (threadIdx.x == 0)
        //    printf("while loop i=%u thread:%d\n", i, threadIdx.x);

        bool found = 0;
        uint32_t u;
        if (threadIdx.x < lsize) {
            int mid, l = 0, r = rn - 1; // [l, r]
            if (verbose)
                printf("th%d fetch u at %d\n", threadIdx.x, i + threadIdx.x);
            u = lbases[i + threadIdx.x]; // u: an element in lbases
            if (verbose)
                printf("th%d u fetched\n", threadIdx.x);
            while (l <= r) {
                mid = (l + r) >> 1;
                if (verbose)
                    printf("th%d l=%d r=%d mid=%d\n", threadIdx.x, l, r, mid);
                if (rbases[mid] < u) {
                    l = mid + 1;
                } else if (rbases[mid] > u) {
                    r = mid - 1;
                } else {
                    found = 1;
                    break;
                }
            }
            // printf("i=%d thread%d u=%d found=%d\n", i, threadIdx.x, u, found);
        }
        tmp_offset[threadIdx.x] = found;
        __syncthreads();

        // currently blockDim.x == THREADS_PER_BLOCK
        for (int s = 1; s < blockDim.x; s *= 2) {
            int index = threadIdx.x;
            if (index >= s) {
                tmp_offset[index] += tmp_offset[index - s];
            }
            __syncthreads();
        }

        //printf("i=%d thread%d found=%d offset=%d\n", i, threadIdx.x, found, tmp_offset[threadIdx.x]);
        if (found) {
            uint32_t offset = tmp_offset[threadIdx.x] - 1;
            tmp[tmp_size + offset] = u;
        }

        if (threadIdx.x == 0) {
            tmp_size += tmp_offset[THREADS_PER_BLOCK - 1];
        }
        __syncthreads();

        i += lsize;
    }
    if (verbose)
        printf("main loop finish th%d\n", threadIdx.x);

    if (threadIdx.x == 0) {
        if (verbose)
            printf("ptr: %p size: %d\n", p_tmp_size, tmp_size);
        *p_tmp_size = tmp_size;
        //printf("tmp_size=%d\n", tmp_size);
    }
}

__device__ uint32_t do_intersection(uint32_t* out, uint32_t* a, uint32_t* b, uint32_t na, uint32_t nb)
{
    __shared__ uint32_t out_offset[THREADS_PER_BLOCK];
    __shared__ uint32_t out_size;

    if (threadIdx.x == 0) {
        out_size = 0;
    }

    uint32_t blk_size, i = 0;
    while (i < na) {
        blk_size = min(na - i, THREADS_PER_BLOCK);

        bool found = 0;
        uint32_t u = 0;
        if (threadIdx.x < blk_size) {
            int mid, l = 0, r = nb - 1; // [l, r]
            u = a[i + threadIdx.x]; // u: an element in set a
            while (l <= r) {
                mid = (l + r) >> 1;
                if (b[mid] < u) {
                    l = mid + 1;
                } else if (b[mid] > u) {
                    r = mid - 1;
                } else {
                    found = 1;
                    // printf("th%d found u=%d\n", threadIdx.x, u);
                    break;
                }
            }
        }
        out_offset[threadIdx.x] = found;
        int num_found = __syncthreads_count(found);

        // currently blockDim.x == THREADS_PER_BLOCK
        for (int s = 1; s < blockDim.x; s *= 2) {
            int index = threadIdx.x;
            if (index >= s) {
                out_offset[index] += out_offset[index - s];
            }
            __syncthreads();
        }

        if (found) {
            uint32_t offset = out_offset[threadIdx.x] - 1;
            out[out_size + offset] = u;
            // printf("th%d write out[%d] = %d;\n", threadIdx.x, out_size + offset, u);
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            out_size += num_found;
        }
        i += blk_size;
    }

    return out_size;
}

__global__ void intersection2(uint32_t *tmp, uint32_t *lbases, uint32_t *rbases, uint32_t ln, uint32_t rn, uint32_t* p_tmp_size)
{
    // make sure ln <= rn
    if (ln > rn) {
        swap(ln, rn);
        swap(lbases, rbases);
    }

    uint32_t intersection_size = do_intersection(tmp, lbases, rbases, ln, rn);
    //printf("th%d size:%d\n", threadIdx.x, intersection_size);

    if (threadIdx.x == 0)
        *p_tmp_size = intersection_size;
}

bool check(uint32_t* c, uint32_t* dev_c, uint32_t nc, uint32_t dev_nc)
{
    if (nc != dev_nc)
        return false;
    for (int i = 0; i < nc; ++i)
        if (c[i] != dev_c[i])
            return false;
    return true;
}

int main()
{
    int T = 100;
    while (T--) {
        uint32_t na, nb, nc, *a, *b, *c;
        na = rand() % 50;
        nb = rand() % 50;
        a = new uint32_t[na];
        b = new uint32_t[nb];
        for (int i = 0; i < na; ++i)
            a[i] = rand() % 100;
        for (int i = 0; i < nb; ++i)
            b[i] = rand() % 100;
        std::sort(a, a + na);
        std::sort(b, b + nb);
        na = std::unique(a, a + na) - a;
        nb = std::unique(b, b + nb) - b;
        c = new uint32_t[std::max(na, nb)];
        nc = std::set_intersection(a, a + na, b, b + nb, c) - c;

        if (na == 0 || nb == 0)
            continue;

        printf("A(%d): ", na);
        for (int i = 0; i < na; ++i)
            printf("%d ", a[i]);
        printf("\nB(%d): ", nb);
        for (int i = 0; i < nb; ++i)
            printf("%d ", b[i]);
        printf("\nC=A&B(%d): ", nc);
        for (int i = 0; i < nc; ++i)
            printf("%d ", c[i]);
        printf("\n");

        uint32_t *dev_a, *dev_b, *dev_c, *dev_nc;
        gpuErrchk(cudaMallocManaged((void**)&dev_a, na * sizeof(uint32_t)));
        gpuErrchk(cudaMallocManaged((void**)&dev_b, nb * sizeof(uint32_t)));
        gpuErrchk(cudaMallocManaged((void**)&dev_c, std::max(na, nb) * sizeof(uint32_t)));
        gpuErrchk(cudaMallocManaged((void**)&dev_nc, sizeof(uint32_t)));
        // printf("%p %p %p %p\n", dev_a, dev_b, dev_c, &dev_nc);

        gpuErrchk(cudaMemcpy(dev_a, a, na * sizeof(uint32_t), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(dev_b, b, nb * sizeof(uint32_t), cudaMemcpyHostToDevice));

        constexpr int num_blocks = 10;
        intersection2<<<1, THREADS_PER_BLOCK>>>(dev_c, dev_a, dev_b, na, nb, dev_nc);
        gpuErrchk(cudaDeviceSynchronize());

        if (check(c, dev_c, nc, *dev_nc))
            printf("OK\n");
        else {
            printf("check failed!\n");
            printf("C[dev(%d)]: ", *dev_nc);
            for (int i = 0; i < *dev_nc; ++i)
                printf("%d ", dev_c[i]);
            printf("\n");
        }

        gpuErrchk(cudaFree(dev_a));
        gpuErrchk(cudaFree(dev_b));
        gpuErrchk(cudaFree(dev_c));
        gpuErrchk(cudaFree(dev_nc));

        delete[] a, b, c;
    }
}
