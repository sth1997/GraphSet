#pragma once

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#define gpuErrchk(ans)                                                                             \
    { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

template <typename T>
__device__ __host__ inline void swap(T &a, T &b) {
    T t(std::move(a));
    a = std::move(b);
    b = std::move(t);
}

struct GPUGroupDim2 {
    int *data;
    int size;
};

struct GPUGroupDim1 {
    GPUGroupDim2 *data;
    int size;
};

struct GPUGroupDim0 {
    GPUGroupDim1 *data;
    int size;
};

#define get_labeled_edge_index(v, label, l, r)                                                     \
    do {                                                                                           \
        int index = v * l_cnt + label;                                                             \
        l = labeled_vertex[index];                                                                 \
        r = labeled_vertex[index + 1];                                                             \
    } while (0)

#define get_edge_index(v, l, r)                                                                    \
    do {                                                                                           \
        l = vertex[v];                                                                             \
        r = vertex[v + 1];                                                                         \
    } while (0)

void dev_alloc_and_copy(void **dst, size_t size, const void *src = nullptr) {
    gpuErrchk(cudaMalloc(dst, size));
    if (src != nullptr) {
        gpuErrchk(cudaMemcpy(*dst, src, size, cudaMemcpyHostToDevice));
    }
}