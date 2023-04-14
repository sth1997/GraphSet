#include <unistd.h>

#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <tuple>
#include <utility>

#include <mpi.h>
#include <omp.h>

#include "dataloader.h"
#include "graph.h"
#include "schedule_IEP.h"

#include "component/gpu_const.cuh"
#include "component/gpu_device_context.cuh"
#include "component/gpu_schedule.cuh"
#include "component/gpu_vertex_set.cuh"
#include "component/utils.cuh"
#include "src/gpu_pattern_matching_static_task.cuh"

#define ForallDevice(i, devices_use, s)                                                                                                              \
    for (int i = 0; i < devices_use; i++) {                                                                                                          \
        cudaSetDevice(i);                                                                                                                            \
        s;                                                                                                                                           \
    }

using TaskStatus = std::tuple<int64_t, int64_t, int64_t>;

constexpr int DEVICE_PER_NODE = 8; // "max" devices per node

unsigned long long gpu_ans = 0;
unsigned long long global_ans = 0;
unsigned long long node_time = 0; // clocks, need to divided by 1e6
int64_t global_time = 0;
int64_t nr_tasks = 0;

cudaEvent_t event[DEVICE_PER_NODE];


void launch_pattern_matching_kernel(PatternMatchingDeviceContext *context, e_index_t total_edge, int no_device, int total_devices, unsigned long long &sum, cudaEvent_t &event) {
    // using std::chrono::system_clock;
    // auto k1 = system_clock::now();
    uint64_t start_edge = 0;
    gpuErrchk(cudaMemcpy(context->dev_cur_edge, &start_edge, sizeof(unsigned long long), cudaMemcpyHostToDevice));
    // gpuErrchk(cudaMemcpy(context->dev_cur_edge, &context->task_start[no_device], sizeof(unsigned long long), cudaMemcpyHostToDevice));
    // log("no_devices %d, task: %d %d\n", no_device, (int)context->task_start[no_device], (int)context->task_start[no_device+1]);
    gpu_pattern_matching_static<<<num_blocks, THREADS_PER_BLOCK, context->block_shmem_size>>>(total_edge, VertexSet::max_intersection_size, chunk_size, context);
    cudaEventRecord(event);
    // gpuErrchk(cudaPeekAtLastError());
    // gpuErrchk(cudaDeviceSynchronize()); 
    // gpuErrchk(cudaMemcpy(&sum, context->dev_sum, sizeof(sum), cudaMemcpyDeviceToHost));
    // auto k2 = system_clock::now();
    // auto kernel_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(k2 - k1);
    // log("Device No. %d, kernel time: %g s\n", no_device, kernel_elapsed.count() / 1e6);
}



void collect_devices_number(int comm_sz, int node, int *recv_buf) {
    int node_devices; 
    gpuErrchk( cudaGetDeviceCount(&node_devices)); 
    node_devices = std::min(node_devices, DEVICE_PER_NODE);
    int *send_buf = new int;
    *send_buf = node_devices;
    MPI_Allgather(send_buf, 1, MPI_INT32_T, recv_buf, 1, MPI_INT32_T, MPI_COMM_WORLD);
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s graph_file [ignored] pattern_string\n", argv[0]);
        return 0;
    }

    Graph *g;
    DataLoader D;
    if (!D.fast_load(g, argv[1])) {
        fprintf(stderr, "Unable to load graph from %s\n", argv[1]);
        return 0;
    }
    g->build_reverse_edges();

    std::string pattern_str = argv[3];
    Pattern p(int(sqrt(pattern_str.length())), pattern_str.c_str());

    bool pattern_valid;
    Schedule_IEP schedule{p, pattern_valid, 1, 1, true, g->v_cnt, g->e_cnt, g->tri_cnt};
    if (!pattern_valid) {
        fprintf(stderr, "invalid pattern %s\n", argv[3]);
        return 0;
    }

    // MPI initialization
    int comm_sz, node;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &node);
    fprintf(stderr, "size = %d rank = %d\n", comm_sz, node);

    // count the total devices for the static task distribution
    int *devices_number = new int[comm_sz];
    collect_devices_number(comm_sz, node, devices_number);
    int total_devices_number = 0, base_device_number = 0;
    for(int i = 0; i < comm_sz; i++) total_devices_number += devices_number[i];
    for(int i = 0; i < node; i++) base_device_number += devices_number[i];

    int node_devices;
    gpuErrchk(cudaGetDeviceCount(&node_devices));
    node_devices = std::min(node_devices, DEVICE_PER_NODE);

    log("Devices count got.\n");

    #pragma omp parallel for
    ForallDevice(i, node_devices,
        gpuErrchk(cudaEventCreate(&event[i]));
    )


    PatternMatchingDeviceContext *context[node_devices];

    unsigned long long sum[node_devices];

    #pragma omp parallel for
    ForallDevice(i, node_devices, 
        gpuErrchk(cudaMallocManaged((void **)&context[i], sizeof(PatternMatchingDeviceContext)));
        context[i]->init(g, schedule, total_devices_number, base_device_number + i);
    )

    log("Context generated.\n");

    using std::chrono::system_clock;
    auto t1 = system_clock::now();

    // #pragma omp parallel num_threads(node_devices)
    // {
    //     int i = omp_get_thread_num();
    //     // printf("hello from thread %d\n", i);
    //     cudaSetDevice(i);
    //     launch_pattern_matching_kernel(context[i], g->e_cnt, base_device_number + i, total_devices_number, sum[i], event[i]);      
    // }

    // for 能跑满八个线程吗？是个问题的。
    // #pragma omp parallel for
    ForallDevice(i, node_devices,
        launch_pattern_matching_kernel(context[i], g->e_cnt, base_device_number + i, total_devices_number, sum[i], event[i]);
    )

    log("Kernels launched.\n");

    // 轮询
    while(true) {
        bool all_finished = 1;
        #pragma unroll
        for(int i = 0; i < node_devices; i++) {
            auto result = cudaEventQuery(event[i]);
            all_finished &= (result != cudaErrorNotReady);
        }
        if(all_finished) break;
    }

    log("All kernel finished.\n");

    // 填答案
    // #pragma omp parallel for
    ForallDevice(i, node_devices,
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaMemcpy(&sum[i], context[i]->dev_sum, sizeof(sum[i]), cudaMemcpyDeviceToHost));
    )

    auto t2 = system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    node_time = elapsed.count();
    for(int i = 0; i < node_devices; i++){
        gpu_ans += sum[i];
    }

    log("node %d receive answer: %lld\n", node, gpu_ans);
    log("node %d time: %.6lf s\n", node, node_time / 1e6);
    
    // reduce answer to root 0
    MPI_Reduce(&gpu_ans, &global_ans, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

    // reduce time to root 0
    MPI_Reduce(&node_time, &global_time, 1, MPI_INT64_T, MPI_MAX, 0, MPI_COMM_WORLD);

    #pragma omp parallel for
    ForallDevice(i, node_devices, 
        context[i]->destroy(); 
        gpuErrchk(cudaFree(context[i]));
    )

    if (node == 0) {
        auto final_ans = (global_ans) / schedule.get_in_exclusion_optimize_redundancy();
        printf("final answer = %ld , final_time = %.6lf s\n", final_ans, global_time / 1e6);
    }
    MPI_Finalize();
    return 0;
}
