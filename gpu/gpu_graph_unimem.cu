#include <dataloader.h>
#include <graph.h>
#include <mpi.h>
#include <omp.h>
#include <schedule_IEP.h>

#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <tuple>
#include <unistd.h>
#include <utility>

#include "component/gpu_const.cuh"
#include "component/gpu_device_context.cuh"
#include "component/gpu_schedule.cuh"
#include "component/gpu_vertex_set.cuh"
#include "component/utils.cuh"

#include "src/gpu_pattern_matching_multidevices.cuh"

#define ForallDevice(i, devices_use, s)                                                                                                              \
    for (int i = 0; i < devices_use; i++) {                                                                                                          \
        cudaSetDevice(i);                                                                                                                            \
        s;                                                                                                                                           \
    }

using TaskStatus = std::tuple<int64_t, int64_t>;

constexpr int DEVICE_PER_NODE = 8; // "max" devices per node
constexpr int NODE_TASK_GRANULARUTY = 640000;
constexpr int INNER_GRANULARUTY = 160000;

constexpr int MSG_BUF_LEN = 5;


TaskStatus task_status;
bool task_ready;

enum MessageType {
    MSG_REQUEST_WORK,  // slave -> master
    MSG_DISPATCH_WORK, // master -> slave
    MSG_REPORT_ANS     // slave -> master
};

uint64_t nr_tasks;
uint64_t global_cur_task; // only scheduler thread of master node will modify this var in working phase
uint64_t global_ans = 0, gpu_ans = 0;
int finished_number = 0;

void process_message(uint64_t recv_buf[], uint64_t send_buf[][MSG_BUF_LEN], int node, int sender) {
    MPI_Request send_req;
    switch (recv_buf[0]) {
    case MSG_REQUEST_WORK: { // me: master
        // recv: NODE_TASK_GRANULARUTY
        // notice: avoid collision
        send_buf[1][0] = MSG_DISPATCH_WORK;
        send_buf[1][1] = global_cur_task;
        send_buf[1][2] = global_cur_task = std::min(global_cur_task + recv_buf[1], nr_tasks);
        MPI_Isend(&send_buf[1][0], 3, MPI_UINT64_T, sender, 0, MPI_COMM_WORLD, &send_req);
        log("master got work request from node %d , replying [%ld, %ld)\n", sender, send_buf[1][1], send_buf[1][2]);
        break;
    }
    case MSG_DISPATCH_WORK: { // me: slave
        // recv: start, end
        uint64_t new_task_cur = recv_buf[1];
        uint64_t new_task_end = recv_buf[2];
        task_status = std::make_tuple(new_task_cur, new_task_end);
        task_ready = true;
        log("slave node %d got task [%llu, %llu)\n", node, new_task_cur, new_task_end);
        break;
    }
    case MSG_REPORT_ANS: { // me: master
        finished_number += 1;
        log("slave node %d finished its work. Finished number: %d\n", sender, finished_number);
        break;
    }
    }
}

// thread 0 is scheduler, communicate with master node
void pattern_matching_mpi(PatternMatchingDeviceContext **context, int node, int node_devices, int comm_sz) {

    static uint64_t recv_buf[MSG_BUF_LEN], send_buf[2][MSG_BUF_LEN];

    MPI_Request send_req, recv_req;
    MPI_Status mpi_status;
    cudaEvent_t event[node_devices];

    using std::chrono::system_clock;
    auto u1 = system_clock::now();

    // #pragma omp parallel for
    ForallDevice(i, node_devices,
        gpuErrchk(cudaEventCreate(&event[i]));
    )

    // node ask for (first) work
    send_buf[0][0] = MSG_REQUEST_WORK;
    send_buf[0][1] = NODE_TASK_GRANULARUTY;
    MPI_Isend(&send_buf[0][0], 2, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD, &send_req);
    log("node %d ask for task.\n", node);


    // create taskItem
    TaskItem *task[node_devices];
    ForallDevice(i, node_devices,
        gpuErrchk( cudaMallocManaged((void**)&task[i], sizeof(TaskItem)) );
        task[i]->init();
    )

    // create lock?

    // launch kernel
    for(int i = 0; i < node_devices; i++) {
        cudaSetDevice(i);
        gpu_pattern_matching_multidevices<<<num_blocks, THREADS_PER_BLOCK, context[i]->block_shmem_size>>>(task[i], VertexSet::max_intersection_size, context[i]);
        cudaEventRecord(event[i]);
    }

    // receive for first message
    MPI_Irecv(recv_buf, MSG_BUF_LEN, MPI_UINT64_T, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &recv_req);
    while (true) {
        int msg_received = 0;
        MPI_Test(&recv_req, &msg_received, &mpi_status);
        while (msg_received) {
            process_message(recv_buf, send_buf, node, mpi_status.MPI_SOURCE);
            msg_received = 0;
            MPI_Irecv(recv_buf, MSG_BUF_LEN, MPI_UINT64_T, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &recv_req);
            MPI_Test(&recv_req, &msg_received, &mpi_status);
        }

        // get new work?
        if(!task_ready) continue;
        unsigned long long start = std::get<0>(task_status), end = std::get<1>(task_status);
        task_ready = false;
        
        // last task?
        if(start == nr_tasks) {
            while (true) {
                bool all_fetched = true;
                ForallDevice(i, node_devices,
                    all_fetched &= (bool)(*(task[i]->task_fetched));
                )
                if(all_fetched) break;
            }

            log("all_fetched.\n");

            // assign a [0, 0] task to all devices
            ForallDevice(i, node_devices,
                *task[i]->new_task_start = 32123;
                *task[i]->new_task_end = 0;
                __sync_fetch_and_and(task[i]->task_fetched, 0);
            )
            
            // break when all devices end kernel function
            while(true) {
                bool all_finished = true;
                ForallDevice(i, node_devices,
                    auto result = cudaEventQuery(event[i]);
                    all_finished &= (result != cudaErrorNotReady);
                )
                if(all_finished) break;
            }
            log("node %d finished.\n", node);
            break;
        }

        // polling and assigning new work

        while(start < end) {
            ForallDevice(i, node_devices,
                // we can give it new task
                if(*task[i]->task_fetched) {
                    log("assign task to devices %d %llu %llu\n", i, start, min(start + INNER_GRANULARUTY, end));
                    *task[i]->new_task_start = start;
                    *task[i]->new_task_end = start = min(start + INNER_GRANULARUTY, end);
                    __sync_fetch_and_and(task[i]->task_fetched, 0);

                //     log("task[%d]->fetched:%d\n",i,(int)*task[i]->task_fetched);
                }
            )
        }

        // node ask for a new job
        send_buf[0][0] = MSG_REQUEST_WORK;
        send_buf[0][1] = NODE_TASK_GRANULARUTY;
        MPI_Isend(&send_buf[0][0], 2, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD, &send_req);
        log("node %d ask for task.\n", node);

    }
    // send "I finished!" to root 
    if(node != 0) {
        send_buf[0][0] = MSG_REPORT_ANS;
        MPI_Isend(send_buf, 1, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD, &send_req);
    }
    // reduce answer
    // Step 1: from multi-devices
    
    unsigned long long sum[node_devices];
    #pragma omp parallel for
    ForallDevice(i, node_devices, 
        gpuErrchk(cudaDeviceSynchronize()); 
        gpuErrchk(cudaMemcpy(&sum[i], context[i]->dev_sum, sizeof(sum[i]), cudaMemcpyDeviceToHost));
    )
    for(int i = 0; i < node_devices; i++){
        gpu_ans += sum[i];
    }
    fprintf(stderr, "node %d receive answer: %lld\n", node, gpu_ans);
    // Step2: reuduce to root 0
    MPI_Reduce(&gpu_ans, &global_ans, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);


    // Destroy task
    ForallDevice(i, node_devices, 
        task[i]->destroy();
        gpuErrchk( cudaFree(task[i]) );
    )
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

    // initialize global work states
    nr_tasks = g->e_cnt;

    int node_devices;
    gpuErrchk( cudaGetDeviceCount(&node_devices));
    node_devices = std::min(node_devices, DEVICE_PER_NODE);

    PatternMatchingDeviceContext *context[node_devices];

    #pragma omp parallel for
    ForallDevice(i, node_devices, 
        gpuErrchk(cudaMallocManaged((void **)&context[i], sizeof(PatternMatchingDeviceContext)));
        context[i]->init(g, schedule);
    )

    using std::chrono::system_clock;
    auto t1 = system_clock::now();

    pattern_matching_mpi(context, node, node_devices, comm_sz);

    auto t2 = system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);

#pragma omp parallel for
    ForallDevice(i, node_devices, 
        context[i]->destroy(); 
        gpuErrchk(cudaFree(context[i]));
    )

    printf("node %d gpu_ans = %ld\n", node, gpu_ans);
    if (node == 0) {
        auto final_ans = (global_ans) / schedule.get_in_exclusion_optimize_redundancy();
        printf("final answer = %ld\n", final_ans);
        printf("time = %g seconds\n", elapsed.count() / 1e6);
    }
    MPI_Finalize();
    return 0;
}
