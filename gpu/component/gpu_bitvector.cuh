#pragma once
#include "gpu_const.cuh"
#include "utils.cuh"

#include <cstdint>
#include <cub/cub.cuh>


class GPUBitVector {
  public:
    GPUBitVector &operator=(const GPUBitVector &) = delete;
    GPUBitVector(const GPUBitVector &&) = delete;
    GPUBitVector(const GPUBitVector &) = delete;

    void construct(size_t element_cnt) {
        size = (element_cnt + 31) / 32;
        gpuErrchk(cudaMalloc((void **)&data, size * sizeof(uint32_t)));
    }
    void destroy() { gpuErrchk(cudaFree(data)); }

    __device__ void clear() {
        non_zero_cnt = 0;
        memset((void *)data, 0, size * sizeof(uint32_t));
    }

    inline __device__ uint32_t &operator[](const int index) { return data[index]; }

    inline __device__ uint32_t calculate_non_zero_cnt() const {
        // warp reduce version
        typedef cub::WarpReduce<uint32_t> WarpReduce;
        __shared__ typename WarpReduce::TempStorage temp_storage[WARPS_PER_BLOCK];
        int wid = threadIdx.x / THREADS_PER_WARP; // warp id
        int lid = threadIdx.x % THREADS_PER_WARP; // lane id
        uint32_t sum = 0;
        for (int index = 0; index < size; index += THREADS_PER_WARP)
            if (index + lid < size)
                sum += __popc(data[index + lid]);
        __syncwarp();
        uint32_t aggregate = WarpReduce(temp_storage[wid]).Sum(sum);
        __syncwarp();
        // brute force version
        // uint32_t aggregate = 0;
        // for(int index = 0; index < size; index++){
        //     aggregate += __popc(data[index]);
        // }
        return aggregate;
    }
    inline __device__ uint32_t get_non_zero_cnt() const { return non_zero_cnt; }

    __device__ void insert(uint32_t id) {
        // data[id >> 5] |= 1 << (id & 31);
        atomicOr(&data[id >> 5], 1 << (id & 31));
        __threadfence_block();
    }
    __device__ uint32_t insert_and_update(uint32_t id) {
        uint32_t index = id >> 5;
        uint32_t tmp_data = data[index];
        uint32_t offset = 1 << (id % 32);
        if ((tmp_data & offset) == 0) {
            ++non_zero_cnt;
            data[index] = tmp_data | offset;
        }
    }
    __host__ __device__ uint32_t *get_data() const { return data; }
    __host__ __device__ size_t get_size() const { return size; }

  private:
    size_t size;
    uint32_t *data;
    uint32_t non_zero_cnt;
};