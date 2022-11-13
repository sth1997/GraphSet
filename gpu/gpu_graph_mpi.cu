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


using TaskStatus = std::tuple<int64_t, int64_t>;
TaskStatus task_status;
SpinLock task_status_lock;

constexpr int NODE_TASK_GRANULARUTY = 5000000;

enum MessageType {
    MSG_REQUEST_WORK,  // slave -> master
    MSG_DISPATCH_WORK, // master -> slave
    MSG_REPORT_ANS     // slave -> master
};

int64_t nr_tasks;
int64_t global_cur_task;       // only scheduler thread of master node will modify this var in working phase
uint64_t global_ans = 0, gpu_ans = 0;

bool task_ready = false;

void process_message(uint64_t recv_buf[], uint64_t send_buf[], int node, int sender) {
    MPI_Request send_req;
    switch (recv_buf[0]) {
    case MSG_REQUEST_WORK: { // me: master
        send_buf[0] = MSG_DISPATCH_WORK;
        send_buf[1] = global_cur_task;
        send_buf[2] = global_cur_task = std::min(global_cur_task + NODE_TASK_GRANULARUTY, nr_tasks);
        MPI_Isend(send_buf, 3, MPI_UINT64_T, sender, 0, MPI_COMM_WORLD, &send_req);
        // log("master got work request from node %d, replying [%ld, %ld)\n", sender, send_buf[1], send_buf[2]);
        break;
    }
    case MSG_DISPATCH_WORK: { // me: slave
        int64_t new_task_cur = recv_buf[1];
        int64_t new_task_end = recv_buf[2];
        LockGuard<SpinLock> guard{task_status_lock};
        task_status = std::make_tuple(new_task_cur, new_task_end);
        task_ready = true;
        // log("slave node %d got task [%lld, %lld)\n", node, new_task_cur, new_task_end);
        break;
    }
    case MSG_REPORT_ANS: { // me: master
        global_ans += recv_buf[1];
        // log("master receive answer %ld from node %d\n", recv_buf[1], sender);
        break;
    }
    }
}

void launch_pattern_matching_kernel(PatternMatchingDeviceContext *context, const TaskStatus &task_range) {
    int64_t task_cur = std::get<0>(task_range);
    int64_t task_end = std::get<1>(task_range);
    // log("task_cur: %lld, task_end: %ld\n", task_cur, task_end);
    // unsigned long long sum = 0;
    // gpuErrchk(cudaMemcpy(context->dev_sum, &sum, sizeof(sum), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(context->dev_cur_edge, &task_cur, sizeof(task_cur), cudaMemcpyHostToDevice));
    gpu_pattern_matching<<<num_blocks, THREADS_PER_BLOCK, context->block_shmem_size>>>(task_end, VertexSet::max_intersection_size, context);
}

// thread 0 is scheduler, communicate with master node
void pattern_matching_mpi(PatternMatchingDeviceContext *context, int node, int comm_sz) {
    bool gpu_working = false, final_task = false;

    constexpr int MSG_BUF_LEN = 256;
    static uint64_t recv_buf[MSG_BUF_LEN], send_buf[MSG_BUF_LEN];
    
    MPI_Request send_req, recv_req;
    MPI_Status mpi_status;
    cudaEvent_t event;
    cudaEventCreate(&event);

    // ask for first task
    send_buf[0] = MSG_REQUEST_WORK;
    MPI_Isend(send_buf, 1, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD, &send_req);
    
    // receive for first task
    MPI_Irecv(recv_buf, MSG_BUF_LEN, MPI_UINT64_T, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &recv_req);
    while (true) {
        int msg_received = 0;
        MPI_Test(&recv_req, &msg_received, &mpi_status);
        while (msg_received) {
            process_message(recv_buf, send_buf, node, mpi_status.MPI_SOURCE);
            MPI_Irecv(recv_buf, MSG_BUF_LEN, MPI_UINT64_T, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &recv_req);
            MPI_Test(&recv_req, &msg_received, &mpi_status);
        }

        // finish all work
        if(!gpu_working && task_ready && std::get<0>(task_status) >= nr_tasks) {
            log("node %d finish all work.\n", node);
            break;
        }

        if (!gpu_working) {
            if(!task_ready)
                continue;

            gpu_working = true;
            launch_pattern_matching_kernel(context, task_status);
            cudaEventRecord(event);
            task_ready = false;
            
            if(std::get<1>(task_status) >= nr_tasks) {
                final_task = true;
            }

            // ask for task, and waiting for resources
            send_buf[0] = MSG_REQUEST_WORK;

            MPI_Isend(send_buf, 1, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD, &send_req);

            log("node %d gpu kernel launched. [%d, %d)\n", node, std::get<0>(task_status), std::get<1>(task_status));
        } else {
            auto result = cudaEventQuery(event);
            if (cudaErrorNotReady == result)
                continue;

            assert(cudaSuccess == result);
            cudaDeviceSynchronize();
            gpu_working = false;

            if (final_task) {
                break;
            }
        }
    }
    unsigned long long sum = 0;
    gpuErrchk(cudaMemcpy(&sum, context->dev_sum, sizeof(sum), cudaMemcpyDeviceToHost));
    log("node %d receive answer: %lld\n", node, sum);
    gpu_ans += sum;

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

    PatternMatchingDeviceContext *context;
    gpuErrchk(cudaMallocManaged((void **)&context, sizeof(PatternMatchingDeviceContext)));
    context->init(g, schedule);

    using std::chrono::system_clock;
    auto t1 = system_clock::now();

    pattern_matching_mpi(context, node, comm_sz);

    auto t2 = system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);

    context->destroy();
    
    printf("node %d gpu_ans = %ld\n", node, gpu_ans);
    if (node == 0) {
        auto final_ans = (global_ans) / schedule.get_in_exclusion_optimize_redundancy();
        printf("final answer = %ld\n", final_ans);
        printf("time = %g seconds\n", elapsed.count() / 1e6);
    }
    MPI_Finalize();
    return 0;
}
