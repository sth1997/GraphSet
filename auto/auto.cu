void pattern_matching_init(Graph *g, const Schedule_IEP& schedule_iep) {
    printf("basic prefix %d, total prefix %d\n", schedule_iep.get_basic_prefix_num(), schedule_iep.get_total_prefix_num());

    int num_blocks = 1024;
    int num_total_warps = num_blocks * WARPS_PER_BLOCK;

    size_t size_edge = g->e_cnt * sizeof(uint32_t);
    size_t size_vertex = (g->v_cnt + 1) * sizeof(uint32_t);
    size_t size_tmp = VertexSet::max_intersection_size * sizeof(uint32_t) * num_total_warps * (schedule_iep.get_total_prefix_num() + 2); //prefix + subtraction + tmp

    schedule_iep.print_schedule();
    uint32_t *edge_from = new uint32_t[g->e_cnt];
    for(uint32_t i = 0; i < g->v_cnt; ++i)
        for(uint32_t j = g->vertex[i]; j < g->vertex[i+1]; ++j)
            edge_from[j] = i;

    tmpTime.check(); 

    uint32_t *dev_edge;
    uint32_t *dev_edge_from;
    uint32_t *dev_vertex;
    uint32_t *dev_tmp;

    gpuErrchk( cudaMalloc((void**)&dev_edge, size_edge));
    gpuErrchk( cudaMalloc((void**)&dev_edge_from, size_edge));
    gpuErrchk( cudaMalloc((void**)&dev_vertex, size_vertex));
    gpuErrchk( cudaMalloc((void**)&dev_tmp, size_tmp));

    gpuErrchk( cudaMemcpy(dev_edge, g->edge, size_edge, cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy(dev_edge_from, edge_from, size_edge, cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy(dev_vertex, g->vertex, size_vertex, cudaMemcpyHostToDevice));

    unsigned long long sum = 0;

    //memcpy schedule
    GPUSchedule* dev_schedule;
    gpuErrchk( cudaMallocManaged((void**)&dev_schedule, sizeof(GPUSchedule)));
    //dev_schedule->transform_in_exclusion_optimize_group_val(schedule);
    int schedule_size = schedule_iep.get_size();
    int max_prefix_num = schedule_size * (schedule_size - 1) / 2;

    int in_exclusion_optimize_vertex_id_size = schedule_iep.in_exclusion_optimize_vertex_id.size();
    int in_exclusion_optimize_array_size  = schedule_iep.in_exclusion_optimize_coef.size();

    assert(in_exclusion_optimize_array_size == schedule_iep.in_exclusion_optimize_coef.size());
    assert(in_exclusion_optimize_array_size == schedule_iep.in_exclusion_optimize_flag.size());

    printf("array size %d\n", in_exclusion_optimize_array_size);
    fflush(stdout);

    int* in_exclusion_optimize_vertex_id = new int[in_exclusion_optimize_vertex_id_size];
    bool* in_exclusion_optimize_vertex_flag = new bool[in_exclusion_optimize_vertex_id_size];
    int* in_exclusion_optimize_vertex_coef = new int[in_exclusion_optimize_vertex_id_size];

    int* in_exclusion_optimize_coef = new int[in_exclusion_optimize_array_size];
    bool* in_exclusion_optimize_flag = new bool[in_exclusion_optimize_array_size];
    int* in_exclusion_optimize_ans_pos = new int[in_exclusion_optimize_array_size];

    for(int i = 0; i < in_exclusion_optimize_vertex_id_size; ++i) {
        in_exclusion_optimize_vertex_id[i] = schedule_iep.in_exclusion_optimize_vertex_id[i];
        in_exclusion_optimize_vertex_flag[i] = schedule_iep.in_exclusion_optimize_vertex_flag[i];
        in_exclusion_optimize_vertex_coef[i] = schedule_iep.in_exclusion_optimize_vertex_coef[i];
    }

    for(int i = 0; i < in_exclusion_optimize_array_size; ++i) {
        in_exclusion_optimize_coef[i] = schedule_iep.in_exclusion_optimize_coef[i];
        in_exclusion_optimize_flag[i] = schedule_iep.in_exclusion_optimize_flag[i];
        in_exclusion_optimize_ans_pos[i] = schedule_iep.in_exclusion_optimize_ans_pos[i];
    }

    gpuErrchk( cudaMallocManaged((void**)&dev_schedule->in_exclusion_optimize_vertex_id, sizeof(int) * in_exclusion_optimize_vertex_id_size));
    gpuErrchk( cudaMemcpy(dev_schedule->in_exclusion_optimize_vertex_id, in_exclusion_optimize_vertex_id, sizeof(int) * in_exclusion_optimize_vertex_id_size, cudaMemcpyHostToDevice));

    gpuErrchk( cudaMallocManaged((void**)&dev_schedule->in_exclusion_optimize_vertex_flag, sizeof(bool) * in_exclusion_optimize_vertex_id_size));
    gpuErrchk( cudaMemcpy(dev_schedule->in_exclusion_optimize_vertex_flag, in_exclusion_optimize_vertex_flag, sizeof(bool) * in_exclusion_optimize_vertex_id_size, cudaMemcpyHostToDevice));

    gpuErrchk( cudaMallocManaged((void**)&dev_schedule->in_exclusion_optimize_vertex_coef, sizeof(int) * in_exclusion_optimize_vertex_id_size));
    gpuErrchk( cudaMemcpy(dev_schedule->in_exclusion_optimize_vertex_coef, in_exclusion_optimize_vertex_coef, sizeof(int) * in_exclusion_optimize_vertex_id_size, cudaMemcpyHostToDevice));

    gpuErrchk( cudaMallocManaged((void**)&dev_schedule->in_exclusion_optimize_coef, sizeof(int) * in_exclusion_optimize_array_size));
    gpuErrchk( cudaMemcpy(dev_schedule->in_exclusion_optimize_coef, in_exclusion_optimize_coef, sizeof(int) * in_exclusion_optimize_array_size, cudaMemcpyHostToDevice));

    gpuErrchk( cudaMallocManaged((void**)&dev_schedule->in_exclusion_optimize_flag, sizeof(bool) * in_exclusion_optimize_array_size));
    gpuErrchk( cudaMemcpy(dev_schedule->in_exclusion_optimize_flag, in_exclusion_optimize_flag, sizeof(bool) * in_exclusion_optimize_array_size, cudaMemcpyHostToDevice));

    gpuErrchk( cudaMallocManaged((void**)&dev_schedule->in_exclusion_optimize_ans_pos, sizeof(int) * in_exclusion_optimize_array_size));
    gpuErrchk( cudaMemcpy(dev_schedule->in_exclusion_optimize_ans_pos, in_exclusion_optimize_ans_pos, sizeof(int) * in_exclusion_optimize_array_size, cudaMemcpyHostToDevice));

    gpuErrchk( cudaMallocManaged((void**)&dev_schedule->adj_mat, sizeof(int) * schedule_size * schedule_size));
    gpuErrchk( cudaMemcpy(dev_schedule->adj_mat, schedule_iep.get_adj_mat_ptr(), sizeof(int) * schedule_size * schedule_size, cudaMemcpyHostToDevice));

    gpuErrchk( cudaMallocManaged((void**)&dev_schedule->father_prefix_id, sizeof(int) * max_prefix_num));
    gpuErrchk( cudaMemcpy(dev_schedule->father_prefix_id, schedule_iep.get_father_prefix_id_ptr(), sizeof(int) * max_prefix_num, cudaMemcpyHostToDevice));

    gpuErrchk( cudaMallocManaged((void**)&dev_schedule->last, sizeof(int) * schedule_size));
    gpuErrchk( cudaMemcpy(dev_schedule->last, schedule_iep.get_last_ptr(), sizeof(int) * schedule_size, cudaMemcpyHostToDevice));

    gpuErrchk( cudaMallocManaged((void**)&dev_schedule->next, sizeof(int) * max_prefix_num));
    gpuErrchk( cudaMemcpy(dev_schedule->next, schedule_iep.get_next_ptr(), sizeof(int) * max_prefix_num, cudaMemcpyHostToDevice));

    gpuErrchk( cudaMallocManaged((void**)&dev_schedule->break_size, sizeof(int) * max_prefix_num));
    gpuErrchk( cudaMemcpy(dev_schedule->break_size, schedule_iep.get_break_size_ptr(), sizeof(int) * max_prefix_num, cudaMemcpyHostToDevice));

    gpuErrchk( cudaMallocManaged((void**)&dev_schedule->loop_set_prefix_id, sizeof(int) * schedule_size));
    gpuErrchk( cudaMemcpy(dev_schedule->loop_set_prefix_id, schedule_iep.get_loop_set_prefix_id_ptr(), sizeof(int) * schedule_size, cudaMemcpyHostToDevice));

    gpuErrchk( cudaMallocManaged((void**)&dev_schedule->restrict_last, sizeof(int) * schedule_size));
    gpuErrchk( cudaMemcpy(dev_schedule->restrict_last, schedule_iep.get_restrict_last_ptr(), sizeof(int) * schedule_size, cudaMemcpyHostToDevice));

    gpuErrchk( cudaMallocManaged((void**)&dev_schedule->restrict_next, sizeof(int) * max_prefix_num));
    gpuErrchk( cudaMemcpy(dev_schedule->restrict_next, schedule_iep.get_restrict_next_ptr(), sizeof(int) * max_prefix_num, cudaMemcpyHostToDevice));

    gpuErrchk( cudaMallocManaged((void**)&dev_schedule->restrict_index, sizeof(int) * max_prefix_num));
    gpuErrchk( cudaMemcpy(dev_schedule->restrict_index, schedule_iep.get_restrict_index_ptr(), sizeof(int) * max_prefix_num, cudaMemcpyHostToDevice));

    dev_schedule->in_exclusion_optimize_array_size = in_exclusion_optimize_array_size;
    dev_schedule->in_exclusion_optimize_vertex_id_size = in_exclusion_optimize_vertex_id_size;
    dev_schedule->size = schedule_iep.get_size();
    dev_schedule->total_prefix_num = schedule_iep.get_total_prefix_num();
    dev_schedule->basic_prefix_num = schedule_iep.get_basic_prefix_num();
    dev_schedule->total_restrict_num = schedule_iep.get_total_restrict_num();
    dev_schedule->in_exclusion_optimize_num = schedule_iep.get_in_exclusion_optimize_num();
    //dev_schedule->k_val = schedule.get_k_val();

    printf("schedule_iep.prefix_num: %d\n", schedule_iep.get_total_prefix_num());
    printf("shared memory for vertex set per block: %ld bytes\n", 
            (schedule_iep.get_total_prefix_num() + 2) * WARPS_PER_BLOCK * sizeof(GPUVertexSet) + in_exclusion_optimize_vertex_id_size * WARPS_PER_BLOCK * sizeof(int));

    tmpTime.print("Prepare time cost");
    tmpTime.check();

    uint32_t buffer_size = VertexSet::max_intersection_size;
    uint32_t block_shmem_size = (schedule_iep.get_total_prefix_num() + 2) * WARPS_PER_BLOCK * sizeof(GPUVertexSet) + in_exclusion_optimize_vertex_id_size * WARPS_PER_BLOCK * sizeof(int);
    //dev_schedule->ans_array_offset = block_shmem_size - in_exclusion_optimize_vertex_id_size * WARPS_PER_BLOCK * sizeof(int);
    // 注意：此处没有错误，buffer_size代指每个顶点集所需的int数目，无需再乘sizeof(uint32_t)，但是否考虑对齐？
    //因为目前用了managed开内存，所以第一次运行kernel会有一定额外开销，考虑运行两次，第一次作为warmup

    int max_active_blocks_per_sm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks_per_sm, gpu_pattern_matching, THREADS_PER_BLOCK, block_shmem_size);
    printf("max number of active warps per SM: %d\n", max_active_blocks_per_sm * WARPS_PER_BLOCK);

    gpu_pattern_matching<<<num_blocks, THREADS_PER_BLOCK, block_shmem_size>>>
        (g->e_cnt, buffer_size, dev_edge_from, dev_edge, dev_vertex, dev_tmp);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk( cudaMemcpyFromSymbol(&sum, dev_sum, sizeof(sum)) );

    //sum /= schedule.get_in_exclusion_optimize_redundancy();

    printf("count %llu\n", sum);
    tmpTime.print("Counting time cost");
    //之后需要加上cudaFree

    // 尝试释放一些内存
    gpuErrchk(cudaFree(dev_edge));
    gpuErrchk(cudaFree(dev_edge_from));
    gpuErrchk(cudaFree(dev_vertex));
    gpuErrchk(cudaFree(dev_tmp));

    gpuErrchk(cudaFree(dev_schedule->adj_mat));
    gpuErrchk(cudaFree(dev_schedule->father_prefix_id));
    gpuErrchk(cudaFree(dev_schedule->last));
    gpuErrchk(cudaFree(dev_schedule->next));
    gpuErrchk(cudaFree(dev_schedule->loop_set_prefix_id));
    gpuErrchk(cudaFree(dev_schedule->restrict_last));
    gpuErrchk(cudaFree(dev_schedule->restrict_next));
    gpuErrchk(cudaFree(dev_schedule->restrict_index));

    gpuErrchk(cudaFree(dev_schedule->in_exclusion_optimize_vertex_id));
    gpuErrchk(cudaFree(dev_schedule->in_exclusion_optimize_coef));
    gpuErrchk(cudaFree(dev_schedule->in_exclusion_optimize_flag));

    gpuErrchk(cudaFree(dev_schedule));

    delete[] edge_from;
    delete[] in_exclusion_optimize_vertex_id;
    delete[] in_exclusion_optimize_coef;
    delete[] in_exclusion_optimize_flag;
}

int main(int argc,char *argv[]) {
    Graph *g;
    DataLoader D;

    /*
       if (argc < 2) {
       printf("Usage: %s dataset_name graph_file [binary/text]\n", argv[0]);
       printf("Example: %s Patents ~hzx/data/patents_bin binary\n", argv[0]);
       printf("Example: %s Patents ~zms/patents_input\n", argv[0]);

       printf("\nExperimental usage: %s [graph_file.g]\n", argv[0]);
       printf("Example: %s ~hzx/data/patents.g\n", argv[0]);
       return 0;
       }

       bool binary_input = false;
       if (argc >= 4)
       binary_input = (strcmp(argv[3], "binary") == 0);

       DataType my_type;
       if (argc >= 3) {
       GetDataType(my_type, argv[1]);

       if (my_type == DataType::Invalid) {
       printf("Dataset not found!\n");
       return 0;
       }
       }*/

    using std::chrono::system_clock;
    auto t1 = system_clock::now();

    bool ok;
    /*
       if (argc >= 3) {
    // 注：load_data的第四个参数用于指定是否读取二进制文件输入，默认为false
    ok = D.load_data(g, my_type, argv[2], binary_input);
    } else {
    ok = D.fast_load(g, argv[1]);
    }
    */

    ok = D.fast_load(g, argv[1]);

    if (!ok) {
        printf("data load failure :-(\n");
        return 0;
    }

    auto t2 = system_clock::now();
    auto load_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    printf("Load data success! time: %g seconds\n", load_time.count() / 1.0e6);
    fflush(stdout);

    allTime.check();

    // const char *pattern_str = "0111010011100011100001100"; // 5 house p1
    //const char *pattern_str = "011011101110110101011000110000101000"; // 6 p2
    // const char *pattern_str = "0111111101111111011101110100111100011100001100000"; // 7 p5
    // const char *pattern_str = "0111111101111111011001110100111100011000001100000"; // 7 p6

    int pattern_size = atoi(argv[2]);
    const char* pattern_str= argv[3];

    Pattern p(pattern_size, pattern_str);
    printf("pattern = \n");
    p.print();
    printf("max intersection size %d\n", VertexSet::max_intersection_size);
    bool is_pattern_valid;
    bool use_in_exclusion_optimize = true;
    Schedule_IEP schedule_iep(p, is_pattern_valid, 1, 1, use_in_exclusion_optimize, g->v_cnt, g->e_cnt, g->tri_cnt);

    if (!is_pattern_valid) {
        printf("pattern is invalid!\n");
        return 0;
    }

    pattern_matching_init(g, schedule_iep);

    allTime.print("Total time cost");

    return 0;
}
