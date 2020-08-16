#include <stdio.h>

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                    msg, cudaGetErrorString(__err), \
                    __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)


const int N=10;

__global__ void add(int *a, int *b, int *c) {
    int tid = threadIdx.x;
    c[tid] = a[tid] + b[tid];
}


int main(){

    int a[N], b[N], c[N];
    int *dev_a, *dev_b, *dev_c;

    cudaMalloc( (void**)&dev_a, N * sizeof(int) );
    cudaMalloc( (void**)&dev_b, N * sizeof(int) );
    cudaMalloc( (void**)&dev_c, N * sizeof(int) );
    cudaCheckErrors("cudamalloc fail");

    for (int i=0; i<N; i++) {
        a[i] = -i; b[i] = i * i;
    }
    cudaMemcpy ( dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice );
    cudaMemcpy ( dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice );
    cudaCheckErrors("cuda memcpy fail");

    add<<<1,N>>>(dev_a, dev_b, dev_c);

    cudaMemcpy(c, dev_c, N * sizeof(int),cudaMemcpyDeviceToHost );
    cudaCheckErrors("cudamemcpy or cuda kernel fail");
    for (int i=0; i<N; i++) {
        printf("%d + %d = %d\n", a[i],b[i],c[i]);
    }

    cudaFree (dev_a); cudaFree (dev_b); cudaFree (dev_c);

    return 0;

}

