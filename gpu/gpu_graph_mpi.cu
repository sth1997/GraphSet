#include <schedule_IEP.h>
#include <mpi.h>
#include <omp.h>

#include <unistd.h>
#include <cstdlib>
#include <cassert>
#include <cstdio>
#include <atomic>
#include <tuple>

#include <utility>

struct SpinLock {
    std::atomic_flag flag;

    SpinLock() : flag{ATOMIC_FLAG_INIT} {}
    void lock() { while (flag.test_and_set()) asm volatile ("pause"); }
    void unlock() { flag.clear(); }
};

template <typename Lock>
struct LockGuard {
    Lock &_lock;
    LockGuard(Lock &lock) : _lock{lock} { _lock.lock(); }
    ~LockGuard() { _lock.unlock(); }
};

__global__ void cuda_kernel(int node)
{
    printf("hello from cuda thread=%d block=%d got rank=%d\n", threadIdx.x, blockIdx.x, node);
}

__global__ void spin_kernel(clock_t cycles)
{
    clock_t start = clock64();
    while (clock64() - start < cycles)
        ;
    printf("spin_kernel done. thread=%d\n", threadIdx.x);
}

// #define log(fmt, ...) fprintf(stderr, fmt, ##__VA_ARGS__)
#define log(fmt, ...) (void)(fmt)

void do_work(int task_id)
{
    // int us = 10000 + rand() % 10000;
    int us = 100 + rand() % 100;
    usleep(us);

    int tid = omp_get_thread_num();
    log("omp thread %d finish task %d\n", tid, task_id);
}

// shared by all threads
int nr_tasks = 10000;
using TaskStatus = std::tuple<int, int>;
TaskStatus task_status;
SpinLock task_status_lock;
volatile bool workers_should_stop = false;
volatile bool task_requested = false;
bool *workers_idle;

constexpr int CPU_WORKER_TASK_GRANULARITY = 2;
constexpr int GPU_WORKER_TASK_GRANULARITY = 5;
constexpr int NODE_TASK_GRANULARUTY = 10;

enum MessageType {
    MSG_REQUEST_WORK,  // slave -> master
    MSG_DISPATCH_WORK, // master -> slave
    MSG_REPORT_ANS     // slave -> master
};

enum NodeState {
    WORKING, // normal working state
    WAITING  // no more work from master node, waiting for worker threads to finish
};

NodeState state = WORKING; // only used by scheduler thread
int global_cur_task; // only scheduler thread of master node will modify this var in working phase
int global_ans = 0, nr_idle_nodes = 0, gpu_ans = 0;
std::atomic<int> node_ans{0};

void process_message(uint64_t recv_buf[], uint64_t send_buf[], int node, int sender)
{
    MPI_Request send_req;
    switch (recv_buf[0]) {
    case MSG_REQUEST_WORK: { // me: master
        send_buf[0] = MSG_DISPATCH_WORK;
        send_buf[1] = global_cur_task;
        send_buf[2] = global_cur_task = std::min(global_cur_task + NODE_TASK_GRANULARUTY, nr_tasks);
        MPI_Isend(send_buf, 3, MPI_UINT64_T, sender, 0, MPI_COMM_WORLD, &send_req);
        log("master got work request from node %d, replying [%ld, %ld)\n", sender, send_buf[1], send_buf[2]);
        break;
    }
    case MSG_DISPATCH_WORK: { // me: slave
        int new_task_cur = recv_buf[1];
        int new_task_end = recv_buf[2];
        if (new_task_cur >= nr_tasks) {
            state = NodeState::WAITING;
            log("slave node %d enters WAITING state\n", node);
        } else {
            LockGuard<SpinLock> guard{task_status_lock};
            task_status = std::make_tuple(new_task_cur, new_task_end);
            log("slave node %d got task [%d, %d)\n", node, new_task_cur, new_task_end);
            task_requested = false;
        }
        break;
    }
    case MSG_REPORT_ANS: { // me: master
        ++nr_idle_nodes;
        global_ans += recv_buf[1];
        log("master receive answer %ld from node %d\n", recv_buf[1], sender);
        break;
    }
    }
}

// TODO: require lock?
bool all_workers_idle()
{
    int nr_threads = omp_get_max_threads();
    int idle_count = 0;
    for (int i = 0; i < nr_threads; ++i)
        if (workers_idle[i])
            ++idle_count;
    return idle_count == nr_threads;
}

// TODO: returns true when too many worker threads are idle?
bool should_request_work()
{
    LockGuard<SpinLock> guard{task_status_lock};
    return std::get<0>(task_status) >= std::get<1>(task_status);
}

// returns whether task status is successfully updated
bool update_task_range(std::tuple<int, int>& task_range, int max_nr_tasks)
{
    int task_cur, task_end, new_task_cur;
    LockGuard<SpinLock> guard{task_status_lock};
    std::tie(task_cur, task_end) = task_status;
    if (task_cur < task_end) {
        new_task_cur = std::min(task_cur + max_nr_tasks, task_end);
        task_range = std::make_tuple(task_cur, new_task_cur);
        task_status = std::make_tuple(new_task_cur, task_end);
        return true;
    }
    return false;
}

// thread 0 is scheduler
void scheduler_loop(int comm_sz, int node)
{
    cudaEvent_t event;
    cudaEventCreate(&event);
    TaskStatus gpu_task_range;
    bool gpu_working = false;

    constexpr int MSG_BUF_LEN = 256;
    static uint64_t recv_buf[MSG_BUF_LEN], send_buf[MSG_BUF_LEN];
    MPI_Request send_req, recv_req;
    MPI_Status mpi_status;

    MPI_Irecv(recv_buf, MSG_BUF_LEN, MPI_UINT64_T, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &recv_req);
    while (true) {
        if (node == 0 || state == NodeState::WORKING) {
            int msg_received = 0;
            MPI_Test(&recv_req, &msg_received, &mpi_status);
            if (msg_received) {
                process_message(recv_buf, send_buf, node, mpi_status.MPI_SOURCE);
                MPI_Irecv(recv_buf, MSG_BUF_LEN, MPI_UINT64_T, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &recv_req);
            }
        }
        
        switch (state) {
        case NodeState::WORKING: {
            if (should_request_work()) {
                if (node != 0) {
                    if (!task_requested) {
                        send_buf[0] = MSG_REQUEST_WORK;
                        MPI_Isend(send_buf, 1, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD, &send_req);
                        task_requested = true;
                    }
                } else {
                    int new_task_cur, new_task_end;
                    new_task_cur = global_cur_task;
                    new_task_end = global_cur_task = std::min(global_cur_task + NODE_TASK_GRANULARUTY, nr_tasks);
                    if (new_task_cur >= nr_tasks) {
                        state = NodeState::WAITING;
                        log("master node enters WAITING state\n");
                    } else {
                        LockGuard<SpinLock> guard{task_status_lock};
                        task_status = std::make_tuple(new_task_cur, new_task_end);
                        log("master node got task [%d, %d)\n", new_task_cur, new_task_end);
                    }
                }
            }
            break;
        }
        case NodeState::WAITING: {
            if (all_workers_idle()) {
                workers_should_stop = true;
                if (node != 0) {
                    send_buf[0] = MSG_REPORT_ANS;
                    send_buf[1] = node_ans;
                    MPI_Isend(send_buf, 2, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD, &send_req);
                    return;
                } else {
                    if (nr_idle_nodes == comm_sz - 1)
                        return;
                }
            }
            break;
        }
        }

        if (!gpu_working) {
            if (update_task_range(gpu_task_range, GPU_WORKER_TASK_GRANULARITY)) {
                // launch cuda kernel
                spin_kernel<<<1, 4>>>(1000000);
                cudaEventRecord(event);
                gpu_working = true;
                workers_idle[0] = false;
                log("node %d gpu kernel launched. [%d, %d)\n", node, std::get<0>(gpu_task_range), std::get<1>(gpu_task_range));
            }
        } else { // poll gpu task state
            auto result = cudaEventQuery(event);
            if (cudaErrorNotReady == result)
                continue;
            
            assert(cudaSuccess == result);
            cudaDeviceSynchronize();
            gpu_working = false;
            workers_idle[0] = true;
            node_ans += std::get<1>(gpu_task_range) - std::get<0>(gpu_task_range);
            gpu_ans  += std::get<1>(gpu_task_range) - std::get<0>(gpu_task_range);
        }
    }
}

// other threads are workers
void worker_loop(int node)
{
    int thread_id = omp_get_thread_num();
    while (!workers_should_stop) {
        TaskStatus task_range;
        bool idle = !update_task_range(task_range, CPU_WORKER_TASK_GRANULARITY);
        if (idle) {
            workers_idle[thread_id] = true;
            continue;
        }
        workers_idle[thread_id] = false;
        
        int task_begin, task_end;
        std::tie(task_begin, task_end) = task_range;
        // for (int i = task_begin; i < task_end; ++i)
        //     do_work(i);
        log("node %d thread %d do work [%d, %d)\n", node, thread_id, task_begin, task_end);
        if (node)
            usleep(100);
        else
            usleep(10000);
        node_ans += task_end - task_begin;
    }
}

void test_cuda_event()
{
    cudaEvent_t event;
    cudaEventCreate(&event);
    spin_kernel<<<1, 32>>>(1000000000); // ~ 1s, 1e9
    cudaEventRecord(event);
    while (true) {
        auto result = cudaEventQuery(event);
        if (cudaSuccess == result)
            break;
        if (cudaErrorNotReady == result) {
            printf("waiting for device...\n");
            usleep(100000);
        } else {
            printf("oops.. %s\n", cudaGetErrorString(result));
            break;
        }
    }
}

int main(int argc, char* argv[])
{
    int comm_sz, node;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &node);
    fprintf(stderr, "size = %d rank = %d\n", comm_sz, node);

    int nr_threads = omp_get_max_threads();
    workers_idle = new bool[nr_threads] {true};
    // init task_status, global_cur_task
    task_status = std::make_tuple(node * 10, node * 10 + 10);
    global_cur_task = comm_sz * 10;

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        if (thread_id == 0) {
            scheduler_loop(comm_sz, node);
        } else {
            worker_loop(node);
        }
        log("node %d thread %d finish.\n", node, thread_id);
    }
    
    MPI_Finalize();
    if (node == 0)
        printf("final answer = %d\n", global_ans + node_ans);
    printf("node %d node_ans = %d gpu_ans = %d\n", node, int(node_ans), gpu_ans);
    return 0;
}
