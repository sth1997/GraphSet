#include <graph.h>
#include <dataloader.h>

#include <cassert>
#include <cstring>
#include <cstdint>
#include <string>
#include <algorithm>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <chrono>
using std::chrono::system_clock;

#include <gpu/set_ops.cuh>

__device__ unsigned long long dev_sum = 0;
__device__ unsigned int dev_cur_edge = 0;
