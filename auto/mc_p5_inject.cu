__global__ void pattern_matching_kernel(uint32_t edge_num, uint32_t buffer_size, uint32_t *edge_from, uint32_t *edge, uint32_t *vertex, uint32_t *tmp) {
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
        subtraction_set.set_data_ptr(block_subtraction_set_buf + wid * 4);

        for (int i = 0; i < 4; ++i) {
            vertex_set[i].set_data_ptr(tmp + offset);
            offset += buffer_size;
        }
    }

    __threadfence_block();

    uint32_t v0, v1, v2, v3;
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
        
        int loop_size_depth2 = vertex_set[1].get_size();
        uint32_t* loop_data_ptr_depth2 = vertex_set[1].get_data_ptr();
        for (int i_depth2 = 0; i_depth2 < loop_size_depth2; ++i_depth2) {
            uint32_t v_depth2 = loop_data_ptr_depth2[i_depth2];
            if (v0 == v_depth2 || v1 == v_depth2) continue;

            unsigned int l_depth2, r_depth2;
            get_edge_index(v_depth2, l_depth2, r_depth2);
            intersection2(vertex_set[2].get_data_ptr(), vertex_set[1].get_data_ptr(), &edge[l_depth2], vertex_set[1].get_size(), r_depth2 - l_depth2, &vertex_set[2].size);
            if (vertex_set[2].get_size() == 0) continue;
            
            v2 = v_depth2; // subtraction_set.push_back(v2);

            int loop_size_depth3 = vertex_set[2].get_size();
            uint32_t* loop_data_ptr_depth3 = vertex_set[2].get_data_ptr();
            for (int i_depth3 = 0; i_depth3 < loop_size_depth3; ++i_depth3) {
                uint32_t v_depth3 = loop_data_ptr_depth3[i_depth3];
                if (v0 == v_depth3 || v1 == v_depth3 || v2 == v_depth3) continue;

                unsigned int l_depth3, r_depth3;
                get_edge_index(v_depth3, l_depth3, r_depth3);
                {
                    tmp_vset = &vertex_set[3];
                    if (threadIdx.x % THREADS_PER_WARP == 0)
                        tmp_vset->init(r_depth3 - l_depth3, &edge[l_depth3]);
                    __threadfence_block();
                    if (r_depth3 - l_depth3 > vertex_set[2].get_size())
                        tmp_vset->size -= unordered_subtraction_size(*tmp_vset, vertex_set[2], -1);
                    else
                        tmp_vset->size = vertex_set[2].get_size() - unordered_subtraction_size(vertex_set[2], *tmp_vset, -1);
                }
                if (vertex_set[3].get_size() == 0) continue;
                
                v3 = v_depth3; // subtraction_set.push_back(v3);

                int ans0 = vertex_set[3].get_size() - 0;
                int ans1 = vertex_set[2].get_size() - 1;
                int ans2 = vertex_set[1].get_size() - 2;
                long long val;
                val = ans0;
                val = val * ans1;
                val = val * ans2;
                sum += val * 1;
                val = ans0;
                val = val * ans1;
                sum += val * -1;
                val = ans0;
                val = val * ans1;
                sum += val * -1;
                val = ans0;
                val = val * ans2;
                sum += val * -1;
                val = ans0;
                sum += val * 2;
            }
        }
    }
    if (lid == 0) atomicAdd(&dev_sum, sum);
}

unsigned long long do_pattern_matching(Graph* g,
    double* p_prepare_time = nullptr, double* p_count_time = nullptr) {
    assert(g != nullptr);
    auto t1 = system_clock::now();

    cudaDeviceProp dev_props;
    cudaGetDeviceProperties(&dev_props, 0);

    int max_active_blocks_per_sm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks_per_sm,
        pattern_matching_kernel, THREADS_PER_BLOCK, 768);
    int nr_blocks = 1024;
    int nr_total_warps = nr_blocks * WARPS_PER_BLOCK;
    printf("nr_blocks=%d\n", nr_blocks);
    
    size_t size_edge = g->e_cnt * sizeof(uint32_t);
    size_t size_vertex = (g->v_cnt + 1) * sizeof(uint32_t);
    size_t size_tmp = VertexSet::max_intersection_size * sizeof(uint32_t) * nr_total_warps * 4;
    uint32_t *edge_from = new uint32_t[g->e_cnt];
    for (uint32_t i = 0; i < g->v_cnt; ++i)
        for (uint32_t j = g->vertex[i]; j < g->vertex[i+1]; ++j)
            edge_from[j] = i;

    uint32_t *dev_edge, *dev_edge_from, *dev_vertex, *dev_tmp;
    gpuErrchk( cudaMalloc((void**)&dev_edge, size_edge));
    gpuErrchk( cudaMalloc((void**)&dev_edge_from, size_edge));
    gpuErrchk( cudaMalloc((void**)&dev_vertex, size_vertex));
    gpuErrchk( cudaMalloc((void**)&dev_tmp, size_tmp));
    gpuErrchk( cudaMemcpy(dev_edge, g->edge, size_edge, cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy(dev_edge_from, edge_from, size_edge, cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy(dev_vertex, g->vertex, size_vertex, cudaMemcpyHostToDevice));

    unsigned long long sum = 0;
    unsigned cur_edge = 0;
    cudaMemcpyToSymbol(dev_sum, &sum, sizeof(sum));
    cudaMemcpyToSymbol(dev_cur_edge, &cur_edge, sizeof(cur_edge));

    auto t2 = system_clock::now();
    double prepare_time = 1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    if (p_prepare_time) *p_prepare_time = prepare_time;
    printf("prepare time: %g seconds\n", prepare_time);
    
    auto t3 = system_clock::now();
    pattern_matching_kernel<<<nr_blocks, THREADS_PER_BLOCK, 768>>>
        (g->e_cnt, VertexSet::max_intersection_size, dev_edge_from, dev_edge, dev_vertex, dev_tmp);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk( cudaMemcpyFromSymbol(&sum, dev_sum, sizeof(sum)) );

    sum /= 2; // IEP redundancy

    auto t4 = system_clock::now();
    double count_time = 1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();
    if (p_count_time) *p_count_time = count_time;
    printf("counting time: %g seconds\n", count_time);
    printf("count: %llu\n", sum);
    
    gpuErrchk(cudaFree(dev_edge));
    gpuErrchk(cudaFree(dev_edge_from));
    gpuErrchk(cudaFree(dev_vertex));
    gpuErrchk(cudaFree(dev_tmp));
    delete[] edge_from;
    return sum;
}
