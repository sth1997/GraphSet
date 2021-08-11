__global__ void gpu_pattern_matching(uint32_t edge_num, uint32_t buffer_size, uint32_t *edge_from, uint32_t *edge, uint32_t *vertex, uint32_t *tmp, const GPUSchedule* schedule) {
    __shared__ unsigned int block_edge_idx[WARPS_PER_BLOCK];
    extern __shared__ GPUVertexSet block_vertex_set[];
    extern __shared__ char block_shmem[];

    int wid = threadIdx.x / THREADS_PER_WARP;
    int lid = threadIdx.x % THREADS_PER_WARP;
    int global_wid = blockIdx.x * WARPS_PER_BLOCK + wid;
    unsigned int &edge_idx = block_edge_idx[wid];
    GPUVertexSet *vertex_set = block_vertex_set + wid * 5;

    GPUVertexSet &subtraction_set = vertex_set[4];
    if (lid == 0) {
        edge_idx = 0;
        uint32_t offset = buffer_size * global_wid * 4;

        uint32_t *block_subtraction_set_buf = (uint32_t *)(block_shmem + 640);
        subtraction_set.set_data_ptr(block_subtraction_set_buf + wid * 6);

        for (int i = 0; i < 4; ++i) {
            vertex_set[i].set_data_ptr(tmp + offset);
            offset += buffer_size;
        }
    }

    __threadfence_block();

    uint32_t v0, v1, v2, v3, v4, v5;
    uint32_t l, r;
    unsigned long long sum = 0;

    while (true) {
        if (lid == 0) {
            edge_idx = atomicAdd(&dev_cur_edge, 1);
        }
        __threadfence_block();

        unsigned int i = edge_idx;
        if (i >= edge_num) break;

        v0 = edge_from[i];
        v1 = edge[i];
        if (v0 <= v1) continue;

        get_edge_index(v0, l, r);
        if (threadIdx.x % THREADS_PER_WARP == 0)
            vertex_set[0].init(r - l, &edge[l]);
        __threadfence_block();
        
        get_edge_index(v1, l, r);
        GPUVertexSet* tmp_vset;
        intersection2(vertex_set[1].get_data_ptr(), vertex_set[0].get_data_ptr(), &edge[l], vertex_set[0].get_size(), r - l, &vertex_set[1].size);
        if (vertex_set[1].get_size() == 0) continue;
        
        if (threadIdx.x % THREADS_PER_WARP == 0)
            vertex_set[2].init(r - l, &edge[l]);
        __threadfence_block();
        if (vertex_set[2].get_size() == 0) continue;
        
        int loop_size_depth2 = vertex_set[1].get_size();
        uint32_t* loop_data_ptr_depth2 = vertex_set[1].get_data_ptr();
        for (int i_depth2 = 0; i_depth2 < loop_size_depth2; ++i_depth2) {
            uint32_t v_depth2 = loop_data_ptr_depth2[i_depth2];
            if (v0 == v_depth2 || v1 == v_depth2) continue;

            unsigned int l_depth2, r_depth2;
            get_edge_index(v_depth2, l_depth2, r_depth2);
            v2 = v_depth2; // subtraction_set.push_back(v2);

            int loop_size_depth3 = vertex_set[1].get_size();
            uint32_t* loop_data_ptr_depth3 = vertex_set[1].get_data_ptr();
            uint32_t min_vertex_depth3 = 0xffffffff;
            min_vertex_depth3 = min(min_vertex_depth3, v2);
            for (int i_depth3 = 0; i_depth3 < loop_size_depth3; ++i_depth3) {
                uint32_t v_depth3 = loop_data_ptr_depth3[i_depth3];
                if (min_vertex_depth3 <= v_depth3) break;
                if (v0 == v_depth3 || v1 == v_depth3 || v2 == v_depth3) continue;

                unsigned int l_depth3, r_depth3;
                get_edge_index(v_depth3, l_depth3, r_depth3);
                v3 = v_depth3; // subtraction_set.push_back(v3);

                int loop_size_depth4 = vertex_set[0].get_size();
                uint32_t* loop_data_ptr_depth4 = vertex_set[0].get_data_ptr();
                for (int i_depth4 = 0; i_depth4 < loop_size_depth4; ++i_depth4) {
                    uint32_t v_depth4 = loop_data_ptr_depth4[i_depth4];
                    if (v0 == v_depth4 || v1 == v_depth4 || v2 == v_depth4 || v3 == v_depth4) continue;

                    unsigned int l_depth4, r_depth4;
                    get_edge_index(v_depth4, l_depth4, r_depth4);
                    intersection2(vertex_set[3].get_data_ptr(), vertex_set[2].get_data_ptr(), &edge[l_depth4], vertex_set[2].get_size(), r_depth4 - l_depth4, &vertex_set[3].size);
                    if (vertex_set[3].get_size() == 1) continue;
                    
                    v4 = v_depth4; // subtraction_set.push_back(v4);

                    int loop_size_depth5 = vertex_set[3].get_size();
                    uint32_t* loop_data_ptr_depth5 = vertex_set[3].get_data_ptr();
                    for (int i_depth5 = 0; i_depth5 < loop_size_depth5; ++i_depth5) {
                        uint32_t v_depth5 = loop_data_ptr_depth5[i_depth5];
                        if (v0 == v_depth5 || v1 == v_depth5 || v2 == v_depth5 || v3 == v_depth5 || v4 == v_depth5) continue;

                        unsigned int l_depth5, r_depth5;
                        get_edge_index(v_depth5, l_depth5, r_depth5);
                        v5 = v_depth5; // subtraction_set.push_back(v5);

                        long long val;
                    }
                }
            }
        }
    }
    if (lid == 0) atomicAdd(&dev_sum, sum);
}
