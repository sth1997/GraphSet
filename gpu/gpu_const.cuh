#pragma once 

constexpr int THREADS_PER_BLOCK = 128;
constexpr int THREADS_PER_WARP = 32;
constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / THREADS_PER_WARP;

constexpr int num_blocks = 1024;
constexpr int num_total_warps = num_blocks * WARPS_PER_BLOCK;
constexpr double cpu_proportion = 0.9;

void print_parameter() {
  printf("THREADS_PER_BLOCK = %d\n", THREADS_PER_BLOCK);
  printf("THREADS_PER_WARP = %d\n", THREADS_PER_WARP);
  printf("num_blocks = %d\n", num_blocks);
  printf("cpu_proportion = %lf\n", cpu_proportion);
}