#pragma once
#include <cstdint>
#include "gpu_const.cuh"
#include "utils.cuh"
#include <cub/cub.cuh>

class GPUBitVector {
public:
    void construct(size_t element_cnt) {
        size = (element_cnt + 31) / 32;
        gpuErrchk( cudaMalloc((void**)&data, size * sizeof(uint32_t)));
    }
    void destroy() {
        gpuErrchk(cudaFree(data));
    }
    __device__ void clear() {
        memset((void*) data, 0, size * sizeof(uint32_t));
    }
    GPUBitVector& operator = (const GPUBitVector&) = delete;
    GPUBitVector(const GPUBitVector&&) = delete;
    GPUBitVector(const GPUBitVector&) = delete;
    inline __device__ uint32_t & operator [] (const int index) { return data[index];}
    
    inline __device__ long long get_non_zero_cnt() const {
        // warp reduce version
        typedef cub::WarpReduce<long long> WarpReduce;
        __shared__ typename WarpReduce::TempStorage temp_storage[WARPS_PER_BLOCK];
        int wid = threadIdx.x / THREADS_PER_WARP; // warp id
        int lid = threadIdx.x % THREADS_PER_WARP; // lane id
        long long sum = 0;
        for(int index = 0; index < size; index += THREADS_PER_WARP) if(index + lid < size)
            sum += __popc(data[index + lid]); 
        __syncwarp();
        long long aggregate = WarpReduce(temp_storage[wid]).Sum(sum);
        __syncwarp();
        // brute force version
        // long long aggregate = 0;
        // for(int index = 0; index < size; index++){
        //     aggregate += __popc(data[index]);
        // }

        return aggregate;
    }
    __device__ void insert(uint32_t id) {
        // data[id >> 5] |= 1 << (id & 31); 
        atomicOr(&data[id >> 5], 1 << (id & 31));
        __threadfence_block();
    }
    __host__ __device__ uint32_t* get_data() const {
        return data;
    }
    __host__ __device__ size_t get_size() const {
        return size;
    }
private:
    size_t size;
    uint32_t* data;
};