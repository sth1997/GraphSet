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

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define THREAD_NUM 32

struct device_pointer {
    int *device_tmp_1;
    int *device_tmp_2;

    int *device_size_1; // size of tmp1
    int *device_size_2; // size of tmp2

    int *device_node; // array of 3 int : v0,v1,v2

    int *device_status;

    unsigned long long *device_ans; // local ans

    int *device_edge;
    unsigned int *device_vertex;
};

Graph *g;

__device__ void swap(int &a, int &b) {
    int t = a;
    a = b;
    b = t;
}

// get the tmp1/2
__global__ void intersection(int *tmp, int *status, unsigned int *vertex, int *edge) {
    int my_rank = blockIdx.x * blockDim.x + threadIdx.x;

    int l0 = status[0];
    int r0 = status[1];

    if(my_rank >= r0 - l0) return;

    int l = status[2];
    int r = status[3];
//    int v0 = status[4];

  //  assert(vertex[v0] == l0);

    int my_id = edge[l0 + my_rank];
    while( l < r - 1) {
        int mid = (l + r) >> 1;
        int cur_id = edge[mid];
        if(my_id >= cur_id) l = mid;
        else r = mid;
    }
    tmp[my_rank] = (my_id == edge[l]) ? 1 : 0; // hit or not
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

__global__ void calculate_ans(unsigned int *pos, unsigned int *e_cnt, int *v_cnt, int *edge, unsigned int *vertex, int *tmp_1, int *tmp_2, int *size_1, int *size_2, int *node, int *status, unsigned long long *ans) {
    unsigned int my_pos = atomicAdd(pos, 1); //which edge is mine
    if(my_pos >= *e_cnt) return;
    int s_l,s_r;
    s_l = 0;
    s_r = *v_cnt;
    while(s_l < s_r - 1) {
        int mid = (s_l + s_r) >> 1;
        if(my_pos >= vertex[mid]) s_l = mid;
        else s_r = mid;
    }

    node[0] = s_l;
    node[1] = edge[my_pos];

    if(node[0] < node[1]) return; // optimize 

    int l0 = vertex[node[0]];
    int r0 = vertex[node[0] + 1];
    int l1 = vertex[node[1]];
    int r1 = vertex[node[1] + 1];
    int v0 = node[0];
    int v1 = node[1];

    if(r0 - l0 > r1 - l1) {
        swap(v0,v1);
        swap(l0,l1);
        swap(r0,r1);
    }

    int BLOCK_NUM = (r0 - l0 + THREAD_NUM - 1) / THREAD_NUM;

    status[0] = l0;
    status[1] = r0;
    status[2] = l1;
    status[3] = r1;
    status[4] = v0;
    status[5] = v1;

    intersection<<<BLOCK_NUM,THREAD_NUM>>>(tmp_1, status, vertex, edge); // intersection N(v0) & N(v1)
    cudaDeviceSynchronize();
    
    *size_1=0;
    for(int pos = 0; pos < r0 - l0; ++pos)
        if(tmp_1[pos]) tmp_1[(*size_1)++] = edge[l0 + pos]; 

    if(*size_1 == 0) return;

    for(int j = vertex[node[0]]; j < vertex[node[0] + 1]; ++j) {
        node[2] = edge[j];
        if(node[1] == node[2]) continue;
        l0 = vertex[node[1]];
        r0 = vertex[node[1] + 1];
        l1 = vertex[node[2]];
        r1 = vertex[node[2] + 1];
        v0 = node[1];
        v1 = node[2];

        if(r0 - l0 > r1 - l1) {
            swap(v0,v1);
            swap(l0,l1);
            swap(r0,r1);
        }

        int BLOCK_NUM = (r0 - l0 + THREAD_NUM - 1) / THREAD_NUM;

        status[0] = l0;
        status[1] = r0;
        status[2] = l1;
        status[3] = r1;
        status[4] = v0;
        status[5] = node[0];

        intersection<<<BLOCK_NUM,THREAD_NUM>>>(tmp_2, status, vertex, edge); // N(v1) & N(v2)
        cudaDeviceSynchronize();
        
        *size_2=0;
        for(int pos = 0; pos < r0 - l0; ++pos)
            if(tmp_2[pos] && edge[l0 + pos] != node[0]) tmp_2[(*size_2)++] = edge[l0 + pos];

        if(*size_2 == 0) continue;

        BLOCK_NUM = ( min(*size_1,*size_2) + THREAD_NUM - 1) / THREAD_NUM; 

        upd_ans<<<BLOCK_NUM,THREAD_NUM>>>(ans, tmp_1, tmp_2, size_1, size_2, node); // ans += tmp1.size * tmp2.size - (tmp1&tmp2).size
        cudaDeviceSynchronize();
    }

}

void gpu_pattern_matching(Graph *g, int thread_count) {

    int size_edge = g->e_cnt;
    int size_vertex = g->v_cnt + 1;
    int size_tmp  = VertexSet::max_intersection_size;

    unsigned long long global_ans = 0;

    unsigned int *device_pos;
    unsigned int *device_e_cnt;
    int *device_v_cnt;

    cudaMalloc((void**)&device_pos, sizeof(int));
    cudaMalloc((void**)&device_e_cnt, sizeof(unsigned int));
    cudaMalloc((void**)&device_v_cnt, sizeof(int));
    cudaMemcpy(device_e_cnt,&g->e_cnt,sizeof(unsigned int),cudaMemcpyHostToDevice);
    cudaMemcpy(device_v_cnt,&g->v_cnt,sizeof(int),cudaMemcpyHostToDevice);
    cudaMemset(device_pos,0,sizeof(unsigned int));

    double t3,t4;
#pragma omp parallel num_threads(thread_count) reduction(+:global_ans)
    {
        printf("%d %d\n",omp_get_thread_num(), omp_get_num_threads());

        cudaStream_t my_stream;
        cudaStreamCreate(&my_stream);

        device_pointer d_ptr;

        cudaMalloc((void**) &d_ptr.device_edge, sizeof(int) * size_edge);
        cudaMalloc((void**) &d_ptr.device_vertex, sizeof(unsigned int) * size_vertex);

        cudaMemcpyAsync(d_ptr.device_edge, g->edge, sizeof(int) * size_edge, cudaMemcpyHostToDevice, my_stream);
        cudaMemcpyAsync(d_ptr.device_vertex, g->vertex, sizeof(unsigned int) * size_vertex, cudaMemcpyHostToDevice, my_stream);

        cudaMalloc((void**) &d_ptr.device_tmp_1, sizeof(int) * size_tmp);
        cudaMalloc((void**) &d_ptr.device_tmp_2, sizeof(int) * size_tmp);

        cudaMalloc((void**) &d_ptr.device_size_1, sizeof(int));
        cudaMalloc((void**) &d_ptr.device_size_2, sizeof(int));

        cudaMalloc((void**) &d_ptr.device_node, sizeof(int) * 3);

        cudaMalloc((void**) &d_ptr.device_status, sizeof(int) * 6);

        cudaMalloc((void**) &d_ptr.device_ans, sizeof(unsigned long long));
        cudaMemset(d_ptr.device_ans, 0, sizeof(unsigned long long));

        if(omp_get_thread_num() == 0) t3 = get_wall_time();

        unsigned long long local_ans;
        unsigned int for_limit = g->e_cnt;
#pragma omp for schedule(dynamic) 
        for(unsigned int i = 0; i < for_limit; ++i) {
            //  printf("for %d %u\n", omp_get_thread_num(), i); fflush(stdout);

            //cudaMemcpyAsync(d_ptr.device_node, &i, sizeof(int), cudaMemcpyHostToDevice, my_stream);

            calculate_ans<<<1,1,0,my_stream>>>(device_pos,device_e_cnt,device_v_cnt,d_ptr.device_edge, d_ptr.device_vertex, d_ptr.device_tmp_1, d_ptr.device_tmp_2, d_ptr.device_size_1, d_ptr.device_size_2, d_ptr.device_node, d_ptr.device_status, d_ptr.device_ans);

            cudaStreamSynchronize(my_stream);
        }
        cudaMemcpyAsync(&local_ans, d_ptr.device_ans, sizeof(unsigned long long), cudaMemcpyDeviceToHost, my_stream);
        cudaStreamSynchronize(my_stream);

        printf("%d ans is %llu\n", omp_get_thread_num(), local_ans);
        global_ans += local_ans;
    }
    t4 = get_wall_time();
    printf("ans %llu\ninner time %.6lf\n", global_ans, t4 - t3);
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
    gpu_pattern_matching(g,24);
    double t2 = get_wall_time();
    printf("time %.6lf\n", t2 - t1);

    delete g;
}
