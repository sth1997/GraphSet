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

#include "src/gpu_pattern_matching.cuh"

#define ForallDevice(i, devices_use, s)                                                                                                              \
    for (int i = 0; i < devices_use; i++) {                                                                                                          \
        cudaSetDevice(i);                                                                                                                            \
        s;                                                                                                                                           \
    }

struct SpinLock {
    std::atomic_flag flag;

    SpinLock() : flag{ATOMIC_FLAG_INIT} {}
    void lock() {
        while (flag.test_and_set())
            asm volatile("pause");
    }
    void unlock() { flag.clear(); }
};

template <typename Lock>
struct LockGuard {
    Lock &_lock;
    LockGuard(Lock &lock) : _lock{lock} { _lock.lock(); }
    ~LockGuard() { _lock.unlock(); }
};

__global__ void spin_kernel(clock_t cycles) {
    clock_t start = clock64();
    while (clock64() - start < cycles)
        ;
    printf("spin_kernel done. thread=%d\n", threadIdx.x);
}


using TaskStatus = std::tuple<int64_t, int64_t, int64_t>;

constexpr int DEVICE_PER_NODE = 8; // "max" devices per node
constexpr int NODE_TASK_GRANULARUTY = 500000 * DEVICE_PER_NODE;// single card

TaskStatus task_status[DEVICE_PER_NODE];
bool is_working[DEVICE_PER_NODE], task_ready[DEVICE_PER_NODE];

enum MessageType {
    MSG_REQUEST_WORK,  // slave -> master
    MSG_DISPATCH_WORK, // master -> slave
    MSG_REPORT_ANS     // slave -> master
};

uint64_t nr_tasks;
uint64_t global_cur_task; // only scheduler thread of master node will modify this var in working phase
uint64_t global_ans = 0, gpu_ans = 0;
int finished_number = 0;

void process_message(uint64_t recv_buf[], uint64_t send_buf[], int node, int sender) {
    MPI_Request send_req;
    switch (recv_buf[0]) {
    case MSG_REQUEST_WORK: { // me: master
        auto send_base = send_buf + 4 * DEVICE_PER_NODE; 
        send_base[0] = MSG_DISPATCH_WORK;
        send_base[1] = recv_buf[1];
        send_base[2] = global_cur_task;
        send_base[3] = global_cur_task = std::min(global_cur_task + recv_buf[2], nr_tasks);
        MPI_Isend(send_base, 4, MPI_UINT64_T, sender, 0, MPI_COMM_WORLD, &send_req);
        log("master got work request from node %d (worker %lld), replying [%ld, %ld)\n", sender, recv_buf[1], send_buf[1], send_buf[2]);
        break;
    }
    case MSG_DISPATCH_WORK: { // me: slave
        int worker = recv_buf[1];
        uint64_t new_task_cur = recv_buf[2];
        uint64_t new_task_end = recv_buf[3];
        task_status[worker] = std::make_tuple(worker, new_task_cur, new_task_end);
        task_ready[worker] = true;
        log("slave node %d (worker %d)  got task [%llu, %llu)\n", node, worker, new_task_cur, new_task_end);
        break;
    }
    case MSG_REPORT_ANS: { // me: master
        finished_number += 1;
        log("slave node %d finished its work. Finished number: %d\n", sender, finished_number);
        break;
    }
    }
}

void launch_pattern_matching_kernel(PatternMatchingDeviceContext *context, const TaskStatus &task_range) {
    int64_t worker = std::get<0>(task_range);
    int64_t task_cur = std::get<1>(task_range);
    int64_t task_end = std::get<2>(task_range);
    log("worker: %d task: [%lld, %lld)\n", worker, task_cur, task_end);
    // unsigned long long sum = 0;
    // gpuErrchk(cudaMemcpy(context->dev_sum, &sum, sizeof(sum), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(context->dev_cur_edge, &task_cur, sizeof(task_cur), cudaMemcpyHostToDevice));
    gpu_pattern_matching<<<num_blocks, THREADS_PER_BLOCK, context->block_shmem_size>>>(task_end, VertexSet::max_intersection_size, context);
}

void ask_for_task(uint64_t *send_base, int worker, MPI_Request &send_req, int node = -1) {
    log("node %d worker %d ask for task.\n", node, worker);
    send_base[0] = MSG_REQUEST_WORK;
    send_base[1] = worker;
    send_base[2] = NODE_TASK_GRANULARUTY;
    MPI_Isend(&send_base, 3, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD, &send_req);
}


// thread 0 is scheduler, communicate with master node
void pattern_matching_mpi(PatternMatchingDeviceContext **context, int node, int node_devices, int comm_sz) {

    constexpr int MSG_BUF_LEN = 256;
    static uint64_t recv_buf[MSG_BUF_LEN], send_buf[MSG_BUF_LEN];

    MPI_Request send_req[node_devices], recv_req;
    MPI_Status mpi_status;
    cudaEvent_t event[node_devices];
    bool finished[node_devices];

    // every worker is ask for task independently

    // #pragma omp parallel for
    ForallDevice(i, node_devices,
        cudaEventCreate(&event[i]);
    )

    for(int i = 0; i < node_devices; i++) { 
        finished[i] = false;
        is_working[i] = false;
        // ask for every first task 
        ask_for_task(send_buf + 4 * i, i, send_req[i], node);
    }

    // receive for first message
    MPI_Irecv(recv_buf, MSG_BUF_LEN, MPI_UINT64_T, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &recv_req);
    while (true) {
        int msg_received = 0;
        MPI_Test(&recv_req, &msg_received, &mpi_status);
        while (msg_received) {
            process_message(recv_buf, send_buf, node, mpi_status.MPI_SOURCE);
            MPI_Irecv(recv_buf, MSG_BUF_LEN, MPI_UINT64_T, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &recv_req);
            MPI_Test(&recv_req, &msg_received, &mpi_status);
        }

        // for all worker
        #pragma omp parallel for
        ForallDevice(i, node_devices,
            if(is_working[i]) {
                // working
                auto result = cudaEventQuery(event[i]);
                // Case 1: result isn't available yet
                if(result == cudaErrorNotReady) continue;
                // Case 2: result is already available, ask for another job
                // use workers' own send buffer[]
                ask_for_task(send_buf + 4 * i, i, send_req[i]);
                is_working[i] = false;
                task_ready[i] = false;
            } else {
                // idle
                // Case 1: task_not ready
                if(!task_ready[i]) continue;
                // Case 2: task is ready, launch kernel
                // Case 2.1: start at final part, there won't be any other task
                if(std::get<1>(task_status[i]) == nr_tasks) {
                    finished[i] = true;
                    continue;
                }
                // do its job anyway
                launch_pattern_matching_kernel(context[i], task_status[i]);
                cudaEventRecord(event[i]);
                is_working[i] = true;
                task_ready[i] = false;
            }
        )
        // Is all devices in a node finished all job?
        bool all_finished = true;
        for(int i = 0; i < node_devices; i++) all_finished &= finished[i];
        // scheduler cannot break, because other nodes may don't know that it's already finished
        if(node != 0 && all_finished) break;
        // scheduler node finish its own work, and all other nodes have finished their work
        if(node == 0 && finished_number == comm_sz - 1 && all_finished) break;
    }
    // send "I finished!" to root 
    if(node != 0) {
        send_buf[0] = MSG_REPORT_ANS;
        MPI_Isend(send_buf, 1, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD, &send_req[0]);
    }
    // reduce answer
    // Step 1: from multi-devices
    
    unsigned long long sum[node_devices];
    #pragma omp parallel for
    ForallDevice(i, node_devices, 
        cudaDeviceSynchronize(); 
        gpuErrchk(cudaMemcpy(&sum[i], context[i]->dev_sum, sizeof(sum[i]), cudaMemcpyDeviceToHost));
    )
    for(int i = 0; i < node_devices; i++){
        gpu_ans += sum[i];
    }
    log("node %d receive answer: %lld\n", node, gpu_ans);
    // Step2: reuduce to root 0
    MPI_Reduce(&gpu_ans, &global_ans, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

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
    cudaGetDeviceCount(&node_devices);
    node_devices = std::min(node_devices, DEVICE_PER_NODE);

    PatternMatchingDeviceContext *context[node_devices];

#pragma omp parallel for
    ForallDevice(i, node_devices, gpuErrchk(cudaMallocManaged((void **)&context[i], sizeof(PatternMatchingDeviceContext)));
                 context[i]->init(g, schedule);)

        using std::chrono::system_clock;
    auto t1 = system_clock::now();

    pattern_matching_mpi(context, node, node_devices, comm_sz);

    auto t2 = system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);

#pragma omp parallel for
    ForallDevice(i, node_devices, context[i]->destroy(); gpuErrchk(cudaFree(context[i]));)

        printf("node %d gpu_ans = %ld\n", node, gpu_ans);
    if (node == 0) {
        auto final_ans = (global_ans) / schedule.get_in_exclusion_optimize_redundancy();
        printf("final answer = %ld\n", final_ans);
        printf("time = %g seconds\n", elapsed.count() / 1e6);
    }
    MPI_Finalize();
    return 0;
}
