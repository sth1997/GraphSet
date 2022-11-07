#pragma once 
#include "component/utils.cuh"

__global__ void kernel_function(int n, int *a, int *b) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if(pos >= n) return;
    a[pos] += b[pos];
}

void test(int n, int *a, int *b) {
    const int BLOCK_NUM = 1000, BLOCK_SIZE = 128;
    int *dev_a, *dev_b;
    gpuErrchk( cudaMalloc((void **)&dev_a, sizeof(int) * n) );
    gpuErrchk( cudaMalloc((void **)&dev_b, sizeof(int) * n) );
    gpuErrchk( cudaMemcpy(dev_a, a, sizeof(int) * n, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(dev_b, b, sizeof(int) * n, cudaMemcpyHostToDevice) );
    kernel_function<<<BLOCK_NUM,BLOCK_SIZE>>>(n,dev_a,dev_b);
    for(int i = 0; i < n; i++) {
        a[i] = a[i] + b[i];
    }

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    gpuErrchk( cudaMemcpy(b, dev_a, sizeof(int) * n, cudaMemcpyDeviceToHost) );

}