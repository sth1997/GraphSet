// This program can only count House pattern using GPU.
#include <../include/graph.h>
#include <../include/dataloader.h>
#include "../include/pattern.h"
#include "../include/schedule.h"
#include "../include/common.h"
#include "../include/motif_generator.h"
#include "../include/vertex_set.h"

#include <assert.h>
#include <iostream>
#include <string>
#include <algorithm>
#include <omp.h>

#define THREAD_NUM 32

struct timer {
    double comb_time;
    double tmp1_time;
    double tmp2_time;
    double get_ans_time;
};

struct device_pointer {
    int *device_tmp_1;
    int *device_tmp_2;

    int *device_size_1;
    int *device_size_2;

    int *device_node;
    int *device_pos;
    int *device_size;
    int *device_cur_depth;

    int *device_status;

    unsigned long long *device_ans;

    int *device_edge;
    unsigned int *device_vertex;
};

struct host_pointer {
    int cur_depth;

    unsigned long long *ans;

    int pos[3], size[3], l, r;

    int v0,v1,v2;
};

Graph *g;

__device__ void swap(int &a, int &b) {
    int t = a;
    a = b;
    b = t;
}

__global__ void init_device_ans(unsigned long long *ans) {
    *ans = 0;
}

__global__ void intersection(int *tmp, int *status, unsigned int *vertex, int *edge) {
    int my_rank = blockIdx.x * blockDim.x + threadIdx.x;

    int l0 = status[0];
    int r0 = status[1];

    if(my_rank >= r0 - l0) return;

    int l = status[2];
    int r = status[3];
    int v0 = status[4];

    int my_id = edge[vertex[v0] + my_rank];
    while( l < r - 1) {
        int mid = (l + r) >> 1;
        int cur_id = edge[mid];
        if(my_id >= cur_id) l = mid;
        else r = mid;
    }
    tmp[my_rank] = (my_id == edge[l]) ? 1 : 0;
    //assert(my_rank == threadIdx.x && my_rank >= 0 && my_rank < 5);
}

__global__ void combining(int *tmp, int *tmp_size, int* status, unsigned int *vertex, int *edge) {
    int limit = status[1] - status[0];
    int v0 = status[4];
    int v1 = status[5];
    *tmp_size = 0;
    for(int i = 0; i < limit; ++i)
        if(tmp[i] && edge[vertex[v0] + i] != v0 && edge[vertex[v0] + i] != v1) tmp[(*tmp_size)] = edge[vertex[v0] + i], *tmp_size += 1;
}

__global__ void get_tmp1_device_status(int *status, int *node, unsigned int *vertex, int *edge) {
    int l0 = vertex[node[0]];
    int r0 = vertex[node[0]+1];
    int l1 = vertex[node[1]];
    int r1 = vertex[node[1]+1];
    int v0 = node[0];
    int v1 = node[1];
    if(r0 - l0 > r1 - l1) {
        swap(v0,v1);
        swap(l0,l1);
        swap(r0,r1);
    }

    status[0] = l0;
    status[1] = r0;
    status[2] = l1;
    status[3] = r1;
    status[4] = v0;
    status[5] = v1;
}

__global__ void get_tmp2_device_status(int *status, int *node, unsigned int *vertex, int *edge) {
    int l0 = vertex[node[1]];
    int r0 = vertex[node[1]+1];
    int l1 = vertex[node[2]];
    int r1 = vertex[node[2]+1];
    int v0 = node[1];
    int v1 = node[2];
    if(r0 - l0 > r1 - l1) {
        swap(v0,v1);
        swap(l0,l1);
        swap(r0,r1);
    }

    status[0] = l0;
    status[1] = r0;
    status[2] = l1;
    status[3] = r1;
    status[4] = v0;
    status[5] = node[0];
}

void get_tmp1(host_pointer &h_ptr, device_pointer &d_ptr, cudaStream_t &my_stream, timer &T) {

    int l0 = g->vertex[h_ptr.v0];
    int r0 = g->vertex[h_ptr.v0 + 1];
    int l1 = g->vertex[h_ptr.v1];
    int r1 = g->vertex[h_ptr.v1 + 1];
    int v0 = h_ptr.v0;
    int v1 = h_ptr.v1;

    if(r0 - l0 > r1 - l1) {
        std::swap(v0,v1);
        std::swap(l0,l1);
        std::swap(r0,r1);
    }

 //   printf("tmp1 %d %d\n", r0 - l0, r1 - l1); 

    int BLOCK_NUM = (r0 - l0 + THREAD_NUM - 1) / THREAD_NUM;

    get_tmp1_device_status<<<1,1,0,my_stream>>>(d_ptr.device_status, d_ptr.device_node, d_ptr.device_vertex, d_ptr.device_edge);
    intersection<<<BLOCK_NUM,THREAD_NUM,0,my_stream>>>(d_ptr.device_tmp_1, d_ptr.device_status, d_ptr.device_vertex, d_ptr.device_edge);
    combining<<<1,1,0,my_stream>>>(d_ptr.device_tmp_1, d_ptr.device_size_1, d_ptr.device_status, d_ptr.device_vertex, d_ptr.device_edge);
}

void get_tmp2(host_pointer &h_ptr, device_pointer &d_ptr, cudaStream_t &my_stream, timer &T) {

    int l0 = g->vertex[h_ptr.v1];
    int r0 = g->vertex[h_ptr.v1 + 1];
    int l1 = g->vertex[h_ptr.v2];
    int r1 = g->vertex[h_ptr.v2 + 1];
    int v0 = h_ptr.v1;
    int v1 = h_ptr.v2;

    if(r0 - l0 > r1 - l1) {
        std::swap(v0,v1);
        std::swap(l0,l1);
        std::swap(r0,r1);
    }
    
 //   printf("tmp2 %d %d\n", r0 - l0, r1 - l1); 

    int BLOCK_NUM = (r0 - l0 + THREAD_NUM - 1) / THREAD_NUM;

    get_tmp2_device_status<<<1,1,0,my_stream>>>(d_ptr.device_status, d_ptr.device_node, d_ptr.device_vertex, d_ptr.device_edge);
    intersection<<<BLOCK_NUM,THREAD_NUM,0,my_stream>>>(d_ptr.device_tmp_2, d_ptr.device_status, d_ptr.device_vertex, d_ptr.device_edge);
    combining<<<1,1,0,my_stream>>>(d_ptr.device_tmp_2, d_ptr.device_size_2, d_ptr.device_status, d_ptr.device_vertex, d_ptr.device_edge);
}

__global__ void add_ans(unsigned long long *ans, int *tmp1, int *tmp2, int *size1, int *size2, int *node) {
    if( *size1 == 0 && *size2 == 0) return;

    int id = node[2];
    int l = 0;
    int r = *size1;
    while( l < r - 1) {
        int mid = (l + r) >> 1;
        if(id >= tmp1[mid]) l = mid;
        else r = mid;
    }
    if( id == tmp1[l]) atomicAdd(ans, (unsigned long long)(*size1 - 1) * (*size2));
    else atomicAdd(ans, (unsigned long long) (*size1) * (*size2));
}

__global__ void upd_ans(unsigned long long *ans, int *tmp1, int *tmp2, int *size1, int *size2, int *node) {
    int my_rank = blockIdx.x * blockDim.x + threadIdx.x;

    if( my_rank >= *size1 || my_rank >= *size2) return;

        if(my_rank == 0) {
          int id = node[2];
          int l = 0;
          int r = *size1;
          while( l < r - 1) {
          int mid = (l + r) >> 1;
          if(id >= tmp1[mid]) l = mid;
          else r = mid;
          }
          if( id == tmp1[l]) atomicAdd(ans, (unsigned long long)(*size1 - 1) * (*size2));
          else atomicAdd(ans, (unsigned long long) (*size1) * (*size2));
          }
     
    if(*size1 < *size2) {
        int my_id = tmp1[my_rank];
        int l = 0;
        int r = *size2;
        while( l < r - 1) {
            int mid = (l + r) >> 1;
            int cur_id = tmp2[mid];
            if(my_id >= cur_id) l = mid;
            else r = mid;
        }
        if( my_id == tmp2[l]) atomicAdd(ans, (unsigned long long)-1);
    }
    else {
        int my_id = tmp2[my_rank];
        int l = 0;
        int r = *size1;
        while( l < r - 1) {
            int mid = (l + r) >> 1;
            int cur_id = tmp1[mid];
            if(my_id >= cur_id) l = mid;
            else r = mid;
        }
        if( my_id == tmp1[l]) atomicAdd(ans, (unsigned long long)-1);
    }
}

void get_ans(host_pointer &h_ptr, device_pointer &d_ptr, cudaStream_t &my_stream, timer &T) {
    int s0 = g->vertex[h_ptr.v0 + 1] - g->vertex[h_ptr.v0];
    int s1 = g->vertex[h_ptr.v1 + 1] - g->vertex[h_ptr.v1];
    int s2 = g->vertex[h_ptr.v2 + 1] - g->vertex[h_ptr.v2];

    int BLOCK_NUM = ( std::min(s0, std::min(s1, s2)) + THREAD_NUM - 1) / THREAD_NUM; // There is a tradeoff : reduce comb is more important, so using size of N(v0,1,2) 

    upd_ans<<<BLOCK_NUM,THREAD_NUM,0,my_stream>>>(d_ptr.device_ans, d_ptr.device_tmp_1, d_ptr.device_tmp_2, d_ptr.device_size_1, d_ptr.device_size_2, d_ptr.device_node);
}

__global__ void status_init(int *node, int *pos, int *size, int *cur_depth, unsigned int *vertex, int *edge) {
    *cur_depth = 0;
    size[1] = size[2] = vertex[node[0] + 1] - vertex[node[0]];
}

__global__ void inc(int *x) {
    *x = *x + 1;
}

__global__ void dec(int *x) {
    *x = *x - 1;
}

void dfs_begin(device_pointer &d_ptr, cudaStream_t &my_stream) {
    inc<<<1,1,0,my_stream>>>(d_ptr.device_cur_depth);
}

void dfs_end(device_pointer &d_ptr, cudaStream_t &my_stream) {
    dec<<<1,1,0,my_stream>>>(d_ptr.device_cur_depth);
}

__global__ void loop_init(int *cur_depth, int *node, int *pos, unsigned int *vertex, int *edge) {
    pos[*cur_depth] = 0;
    node[*cur_depth] = edge[vertex[node[0]]];
}

void for_loop_init(device_pointer &d_ptr, cudaStream_t &my_stream) {
    loop_init<<<1,1,0,my_stream>>>(d_ptr.device_cur_depth, d_ptr.device_node, d_ptr.device_pos, d_ptr.device_vertex, d_ptr.device_edge);
}

__global__ void loop_inc(int *cur_depth, int *node, int *pos, int *size, unsigned int *vertex, int *edge) {
    pos[*cur_depth]++;
    if(pos[*cur_depth] < size[*cur_depth]) {
        node[*cur_depth] = edge[vertex[node[0]] + pos[*cur_depth]];
    }
}

void for_loop_inc(device_pointer &d_ptr, cudaStream_t &my_stream) {
    loop_inc<<<1,1,0,my_stream>>>(d_ptr.device_cur_depth, d_ptr.device_node, d_ptr.device_pos, d_ptr.device_size, d_ptr.device_vertex, d_ptr.device_edge);
}

void dfs(int cur_depth, int thread_count, host_pointer &h_ptr, device_pointer &d_ptr, cudaStream_t &my_stream, timer &T) {
    if( cur_depth > 0) {
        dfs_begin(d_ptr,my_stream);
        for(h_ptr.pos[cur_depth] = 0, for_loop_init(d_ptr,my_stream); h_ptr.pos[cur_depth] < h_ptr.size[cur_depth]; ++h_ptr.pos[cur_depth], for_loop_inc(d_ptr,my_stream)) {

            if( cur_depth == 1) {
                h_ptr.v1 = g->edge[g->vertex[h_ptr.v0] + h_ptr.pos[1]];

                get_tmp1(h_ptr, d_ptr, my_stream, T);

                h_ptr.size[2] = h_ptr.r - h_ptr.l;
            }

            if( cur_depth == 2) {
                if(h_ptr.pos[2] == h_ptr.pos[1]) continue;
                h_ptr.v2 = g->edge[g->vertex[h_ptr.v0] + h_ptr.pos[2]];

                get_tmp2(h_ptr, d_ptr, my_stream, T);

                get_ans(h_ptr, d_ptr, my_stream, T);
                continue;
            }

            dfs(cur_depth + 1, thread_count, h_ptr, d_ptr, my_stream, T);
        }
        dfs_end(d_ptr,my_stream);
    }
    else {
        int for_limit = h_ptr.size[0];
#pragma omp for schedule(dynamic) 
        for(int i = 0; i < for_limit; ++i) {
            printf("for %d %d\n", omp_get_thread_num(), i); fflush(stdout);

            h_ptr.pos[0] = i;
            h_ptr.v0 = h_ptr.pos[cur_depth];
            h_ptr.l = g->vertex[h_ptr.pos[cur_depth]];
            h_ptr.r = g->vertex[h_ptr.pos[cur_depth] + 1];
            h_ptr.size[1] = h_ptr.r - h_ptr.l;

            int v0 = i;
            cudaMemcpyAsync(d_ptr.device_node, &v0, sizeof(int), cudaMemcpyHostToDevice, my_stream);
            status_init<<<1,1,0,my_stream>>>(d_ptr.device_node, d_ptr.device_pos, d_ptr.device_size, d_ptr.device_cur_depth, d_ptr.device_vertex, d_ptr.device_edge);

            dfs(cur_depth + 1, thread_count, h_ptr, d_ptr, my_stream, T);
        }
    }

}

void gpu_pattern_matching(Graph *g, int thread_count) {

    int size_edge = g->e_cnt;
    int size_vertex = g->v_cnt + 1;
    int size_tmp  = VertexSet::max_intersection_size;

    long long global_ans = 0;
    double tmp1_time = 0.0;
    double tmp2_time = 0.0;
    double comb_time = 0.0;
    double get_ans_time = 0.0;


#pragma omp parallel num_threads(thread_count) reduction(+:global_ans) reduction(+:tmp1_time) reduction(+:tmp2_time) reduction(+:comb_time) reduction(+:get_ans_time)
    {
        printf("%d %d\n",omp_get_thread_num(), omp_get_num_threads());
        cudaStream_t my_stream;
        cudaStreamCreate(&my_stream);

        timer T;
        T.tmp1_time = T.tmp2_time = T.comb_time = T.get_ans_time = 0;

        device_pointer d_ptr;
        host_pointer h_ptr;

        cudaMalloc((void**) &d_ptr.device_edge, sizeof(int) * size_edge);
        cudaMalloc((void**) &d_ptr.device_vertex, sizeof(unsigned int) * size_vertex);

        cudaMemcpyAsync(d_ptr.device_edge, g->edge, sizeof(int) * size_edge, cudaMemcpyHostToDevice, my_stream);
        cudaMemcpyAsync(d_ptr.device_vertex, g->vertex, sizeof(unsigned int) * size_vertex, cudaMemcpyHostToDevice, my_stream);

        h_ptr.ans = new unsigned long long(0);

        cudaMalloc((void**) &d_ptr.device_tmp_1, sizeof(int) * size_tmp);
        cudaMalloc((void**) &d_ptr.device_tmp_2, sizeof(int) * size_tmp);

        cudaMalloc((void**) &d_ptr.device_size_1, sizeof(int));
        cudaMalloc((void**) &d_ptr.device_size_2, sizeof(int));

        cudaMalloc((void**) &d_ptr.device_node, sizeof(int) * 3);
        cudaMalloc((void**) &d_ptr.device_pos, sizeof(int) * 3);
        cudaMalloc((void**) &d_ptr.device_size, sizeof(int) * 3);
        cudaMalloc((void**) &d_ptr.device_cur_depth, sizeof(int));

        cudaMalloc((void**) &d_ptr.device_status, sizeof(int) * 6);

        cudaMalloc((void**) &d_ptr.device_ans, sizeof(unsigned long long));

        init_device_ans<<<1,1,0,my_stream>>>(d_ptr.device_ans);

        h_ptr.size[0] = g->v_cnt;
        dfs(0, thread_count, h_ptr, d_ptr, my_stream, T);

        cudaMemcpyAsync(h_ptr.ans, d_ptr.device_ans, sizeof(unsigned long long), cudaMemcpyDeviceToHost, my_stream);
        cudaStreamSynchronize(my_stream);

        global_ans += *h_ptr.ans;
/*      tmp1_time += T.tmp1_time;
        tmp2_time += T.tmp2_time;
        comb_time += T.comb_time;
        get_ans_time += T.get_ans_time;*/
    }
    printf("ans %lld\n", global_ans);
    printf("tmp1 %.6lf\n", tmp1_time);
    printf("tmp2 %.6lf\n", tmp2_time);
    printf("comb %.6lf\n", comb_time);
    printf("get_ans %.6lf\n", get_ans_time);
}

int main(int argc,char *argv[]) {
    DataLoader D;
    /*    
       const std::string type = argv[1];
       const std::string path = argv[2];

       DataType my_type;

       GetDataType(my_type, type);

       if(my_type == DataType::Invalid) {
       printf("Dataset not found!\n");
       return 0;
       }
     */
    //assert(D.load_data(g,my_type,path.c_str())==true); 
    assert(D.load_data(g,100));
    printf("Load data success!\n");
    fflush(stdout);

    double t1 = get_wall_time();
    gpu_pattern_matching(g,48);
    double t2 = get_wall_time();
    printf("time %.6lf\n", t2 - t1);

    delete g;
}
