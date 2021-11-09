#include <cstdio>
#include <cstdlib>
#include <cassert>

#include <graph.h>
#include <dataloader.h>
#include <schedule_IEP.h>

#include <gpu/config.cuh>
#include <gpu/vertex_set.cuh>

#include <unistd.h>

#define write_code(fmt, ...) do { \
    for (int _ = 0; _ < indentation; ++_) printf(" "); \
    printf(fmt, ## __VA_ARGS__); \
} while (0)

#define write_code_noindent printf

void emit_build_vertex_set(const Schedule_IEP& schedule, int prefix_id, int indentation, const char* input_data_str, const char* input_size_str)
{
    int father_id = schedule.get_father_prefix_id(prefix_id);
    if (father_id == -1) {
        write_code("if (threadIdx.x %% THREADS_PER_WARP == 0)\n");
        write_code("    vertex_set[%d].init(%s, %s);\n", prefix_id, input_size_str, input_data_str);
        write_code("__threadfence_block();\n");
    } else {
        bool only_need_size = schedule.get_prefix_only_need_size(prefix_id);

        static bool first_time = true;
        if (first_time) {
            write_code("GPUVertexSet* tmp_vset;\n");
            first_time = false;
        }

        if (only_need_size) {
            write_code("{\n");
            indentation += 4;
            write_code("tmp_vset = &vertex_set[%d];\n", prefix_id);
            write_code("if (threadIdx.x %% THREADS_PER_WARP == 0)\n");
            write_code("    tmp_vset->init(%s, %s);\n", input_size_str, input_data_str);
            write_code("__threadfence_block();\n");

            // write_code("tmp_vset->size = get_intersection_size(*tmp_vset, vertex_set[%d]);\n", father_id);
            write_code("if (%s > vertex_set[%d].get_size())\n", input_size_str, father_id);
            write_code("    tmp_vset->size -= unordered_subtraction_size(*tmp_vset, vertex_set[%d], -1);\n", father_id);
            write_code("else\n");
            write_code("    tmp_vset->size = vertex_set[%d].get_size() - unordered_subtraction_size(vertex_set[%d], *tmp_vset, -1);\n", father_id, father_id);
            indentation -= 4;
            write_code("}\n");
        } else {
            write_code(
                "intersection2(vertex_set[%d].get_data_ptr(), vertex_set[%d].get_data_ptr(), %s, vertex_set[%d].get_size(), %s, &vertex_set[%d].size);\n",
                prefix_id, father_id, input_data_str, father_id, input_size_str, prefix_id
            );
        }
    }
}

void emit_pattern_matching_kernel(const Schedule_IEP& schedule)
{
    int iepo_vertex_id_size = schedule.in_exclusion_optimize_vertex_id.size();
    int iepo_array_size = schedule.in_exclusion_optimize_coef.size();

    assert(iepo_array_size == (int) schedule.in_exclusion_optimize_flag.size());

    int indentation = 0;

    write_code("__global__ void pattern_matching_kernel(uint32_t edge_num, uint32_t buffer_size, uint32_t *edge_from, uint32_t *edge, uint32_t *vertex, uint32_t *tmp) {\n");
    indentation += 4;
    write_code("__shared__ unsigned int block_edge_idx[WARPS_PER_BLOCK];\n");
    write_code("extern __shared__ GPUVertexSet block_vertex_set[];\n");
    write_code("extern __shared__ char block_shmem[];\n\n");

    int num_prefixes = schedule.get_total_prefix_num();
    int num_vertex_sets_per_warp = num_prefixes + 1;

    write_code("int wid = threadIdx.x / THREADS_PER_WARP;\n");
    write_code("int lid = threadIdx.x %% THREADS_PER_WARP;\n");
    write_code("int global_wid = blockIdx.x * WARPS_PER_BLOCK + wid;\n");
    write_code("unsigned int &edge_idx = block_edge_idx[wid];\n");
    write_code("GPUVertexSet *vertex_set = block_vertex_set + wid * %d;\n\n", num_vertex_sets_per_warp);

    write_code("GPUVertexSet &subtraction_set = vertex_set[%d];\n", num_prefixes);
    // write_code("GPUVertexSet &tmp_set = vertex_set[%d];\n", num_prefixes + 1);

    write_code("if (lid == 0) {\n");
    indentation += 4;
    write_code("edge_idx = 0;\n");
    write_code("uint32_t offset = buffer_size * global_wid * %d;\n\n", num_prefixes);

    int block_subtraction_set_offset = num_vertex_sets_per_warp * WARPS_PER_BLOCK * sizeof(GPUVertexSet);
    write_code("uint32_t *block_subtraction_set_buf = (uint32_t *)(block_shmem + %d);\n", block_subtraction_set_offset);
    write_code("subtraction_set.set_data_ptr(block_subtraction_set_buf + wid * %d);\n\n", schedule.get_size() - schedule.get_in_exclusion_optimize_num());

    write_code("for (int i = 0; i < %d; ++i) {\n", num_prefixes);
    indentation += 4;
    write_code("vertex_set[i].set_data_ptr(tmp + offset);\n");
    write_code("offset += buffer_size;\n");
    indentation -= 4;
    write_code("}\n");
    indentation -= 4;
    write_code("}\n\n");

    write_code("__threadfence_block();\n\n");

    write_code("uint32_t v0, v1");
    // 直接以局部变量代替原先的subtraction_set
    for (int depth = 2; depth < schedule.get_size() - schedule.get_in_exclusion_optimize_num(); ++depth)
        write_code_noindent(", v%d", depth);
    write_code_noindent(";\n");
    write_code("uint32_t l, r;\n");

    write_code("unsigned long long sum = 0;\n\n");

    write_code("while (true) {\n");
    indentation += 4;
    write_code("if (lid == 0) {\n");
    indentation += 4;
    write_code("edge_idx = atomicAdd(&dev_cur_edge, 1);\n");
    // write_code("unsigned int i = edge_idx;\n");
    // write_code("if (i < edge_num) {\n");
    // indentation += 4;
    // write_code("subtraction_set.init();\n");
    // write_code("subtraction_set.push_back(edge_from[i]);\n");
    // write_code("subtraction_set.push_back(edge[i]);\n");
    // indentation -= 4;
    // write_code("}\n");
    indentation -= 4;
    write_code("}\n");

    write_code("__threadfence_block();\n\n");

    write_code("unsigned int i = edge_idx;\n");
    write_code("if (i >= edge_num) break;\n\n");
       
    write_code("v0 = edge_from[i];\n");
    write_code("v1 = edge[i];\n");

    if (schedule.get_restrict_last(1) != -1) {
        write_code("if (v0 <= v1) continue;\n\n");
    } 

    // write_code("bool is_zero = false;\n");
    write_code("get_edge_index(v0, l, r);\n");
    for (int prefix_id = schedule.get_last(0); prefix_id != -1; prefix_id = schedule.get_next(prefix_id)) {
        emit_build_vertex_set(schedule, prefix_id, indentation, "&edge[l]", "r - l");
        write_code("\n");
    }

    write_code("get_edge_index(v1, l, r);\n");
    for (int prefix_id = schedule.get_last(1); prefix_id != -1; prefix_id = schedule.get_next(prefix_id)) {
        // write_code("vertex_set[%d].build_vertex_set(%d, vertex_set, &edge[l], r - l);\n", prefix_id, schedule.get_father_prefix_id(prefix_id));
        emit_build_vertex_set(schedule, prefix_id, indentation, "&edge[l]", "r - l");
        if (prefix_id < schedule.get_basic_prefix_num())
            write_code("if (vertex_set[%d].get_size() == 0) continue;\n", prefix_id); //因为代码生成后没有外层的for循环了，所以不需要先break再continue了
        write_code("\n");    
    }
        
    // write_code("extern __shared__ char ans_array[];\n");
    // write_code("int* ans = ((int*) (ans_array + %d)) + %d * (threadIdx.x / THREADS_PER_WARP);\n\n", schedule.ans_array_offset, schedule.in_exclusion_optimize_vertex_id_size);
        
    for (int depth = 2; depth < schedule.get_size() - schedule.get_in_exclusion_optimize_num();  ++depth) {
        int loop_set_prefix_id = schedule.get_loop_set_prefix_id(depth);
        write_code("int loop_size_depth%d = vertex_set[%d].get_size();\n", depth, loop_set_prefix_id);
        // write_code("if (loop_size_depth%d <= 0) continue;\n", depth);

        write_code("uint32_t* loop_data_ptr_depth%d = vertex_set[%d].get_data_ptr();\n",depth, loop_set_prefix_id);
            
        int restrict_depth = schedule.get_restrict_last(depth);
        bool has_min_vertex = restrict_depth != -1;
        if (has_min_vertex) {
            write_code("uint32_t min_vertex_depth%d = v%d;\n", depth, schedule.get_restrict_index(restrict_depth));
        }

        for (int i = schedule.get_restrict_next(restrict_depth); i != -1 && has_min_vertex; i = schedule.get_restrict_next(i)) {
            // write_code("if(min_vertex_depth%d > subtraction_set.get_data(%d)) min_vertex_depth%d = subtraction_set.get_data(%d);\n", depth, schedule.get_restrict_index(i), depth, schedule.get_restrict_index(i));
            write_code("min_vertex_depth%d = min(min_vertex_depth%d, v%d);\n", depth, depth, schedule.get_restrict_index(i));
        }

        write_code("for (int i_depth%d = 0; i_depth%d < loop_size_depth%d; ++i_depth%d) {\n", depth, depth, depth, depth);
        indentation += 4;
        write_code("uint32_t v_depth%d = loop_data_ptr_depth%d[i_depth%d];\n", depth, depth, depth);
        if (has_min_vertex) {
            write_code("if (min_vertex_depth%d <= v_depth%d) break;\n", depth, depth);
        }

        // write_code("if (subtraction_set.has_data(v_depth%d)) continue;\n", depth);
        write_code("if (v0 == v_depth%d || v1 == v_depth%d", depth, depth);
        for (int i = 2; i < depth; ++i)
            write_code_noindent(" || v%d == v_depth%d", i, depth);
        write_code_noindent(") continue;\n\n");

        write_code("unsigned int l_depth%d, r_depth%d;\n", depth, depth);
        write_code("get_edge_index(v_depth%d, l_depth%d, r_depth%d);\n", depth, depth, depth);

        for (int prefix_id = schedule.get_last(depth); prefix_id != -1; prefix_id = schedule.get_next(prefix_id)) {
            char data_str[64] = "&edge[l_depth?]";
            data_str[13] = depth + '0';
            char size_str[64] = "r_depth? - l_depth?";
            size_str[7] = depth + '0';
            size_str[18] = depth + '0';
            //sprintf(data_str, "&edge[l_depth%d]", depth);
            //sprintf(size_str, "r_depth%d - l_depth%d", depth, depth);
            emit_build_vertex_set(schedule, prefix_id, indentation, data_str, size_str);
            if (prefix_id < schedule.get_basic_prefix_num()) //在general版本中没有这个判断，是因为general加了这个判断会更慢（不如直接判断break_size），但在这个版本加了这个判断可能会省去之后的break_size判断
                write_code("if (vertex_set[%d].get_size() == %d) continue;\n", prefix_id, schedule.get_break_size_ptr()[prefix_id]);
            write_code("\n");
        }

        // write_code("if (threadIdx.x % THREADS_PER_WARP == 0) subtraction_set.push_back(v_depth%d);\n", depth);
        // write_code("__threadfence_block();\n\n");
        write_code("v%d = v_depth%d; // subtraction_set.push_back(v%d);\n\n", depth, depth, depth);
        // 最后一个是多余的，我们期望编译器来把它优化掉（但会产生warning）
    }
        
    bool subtraction_set_required = false;
    for (int i = 0; i < iepo_vertex_id_size; ++i) {
        if (!schedule.in_exclusion_optimize_vertex_flag[i]) {
            subtraction_set_required = true;
            break;
        }
    }
    if (subtraction_set_required) {
        // build subtraction set
        write_code("if (lid == 0) {\n");
        indentation += 4;
        write_code("uint32_t *p = subtraction_set.get_data_ptr();\n");
        for (int i = 0; i < schedule.get_size() - schedule.get_in_exclusion_optimize_num(); ++i)
            write_code("p[%d] = v%d;\n", i, i);
        write_code("subtraction_set.set_size(%d);\n", schedule.get_size() - schedule.get_in_exclusion_optimize_num());
        indentation -= 4;
        write_code("}\n");
        write_code("__threadfence_block();\n\n");
    }

    for (int i = 0; i < iepo_vertex_id_size; ++i) {
        if (schedule.in_exclusion_optimize_vertex_flag[i]) {
            write_code("int ans%d = vertex_set[%d].get_size() - %d;\n", i, schedule.in_exclusion_optimize_vertex_id[i], schedule.in_exclusion_optimize_vertex_coef[i]);
        } else {
            write_code("int ans%d = unordered_subtraction_size(vertex_set[%d], subtraction_set);\n", i, schedule.in_exclusion_optimize_vertex_id[i]);
        }
    }

    if (iepo_array_size) {
        int last_pos = -1;
        write_code("long long val;\n");
        for (int pos = 0; pos < iepo_array_size; ++pos) {
            if (pos == last_pos + 1) {
                write_code("val = ans%d;\n", schedule.in_exclusion_optimize_ans_pos[pos]);
            } else {
                write_code("val = val * ans%d;\n", schedule.in_exclusion_optimize_ans_pos[pos]);
            }
            if (schedule.in_exclusion_optimize_flag[pos]) {
                last_pos = pos;
                write_code("sum += val * %d;\n", schedule.in_exclusion_optimize_coef[pos]);
            }
        }
    } else {
        write_code("++sum;\n");
    }

    for (int depth = schedule.get_size() - schedule.get_in_exclusion_optimize_num() - 1; depth >= 2; --depth) {
        // write_code("if (threadIdx.x % THREADS_PER_WARP == 0) subtraction_set.pop_back();\n");
        // write_code("__threadfence_block();\n");

        indentation -= 4;
        write_code("}\n");
    }
    indentation -= 4;
    write_code("}\n");
    write_code("if (lid == 0) atomicAdd(&dev_sum, sum);\n");
    indentation -= 4;
    write_code("}\n\n");
}

void emit_kernel_launch(const Schedule_IEP& schedule, bool verbose = true)
{
    int indentation = 0;

    write_code("unsigned long long do_pattern_matching(Graph* g,\n");
    write_code("    double* p_prepare_time = nullptr, double* p_count_time = nullptr) {\n");
    indentation += 4;

    write_code("assert(g != nullptr);\n");
    write_code("auto t1 = system_clock::now();\n\n");

    size_t block_subtraction_set_size = (schedule.get_size() - schedule.get_in_exclusion_optimize_num()) * WARPS_PER_BLOCK * sizeof(uint32_t);
    size_t block_shmem_size = (schedule.get_total_prefix_num() + 1) * WARPS_PER_BLOCK * sizeof(GPUVertexSet) + block_subtraction_set_size;

    write_code("cudaDeviceProp dev_props;\n");
    write_code("cudaGetDeviceProperties(&dev_props, 0);\n\n");

    write_code("int max_active_blocks_per_sm;\n");
    write_code("cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks_per_sm,\n");
    write_code("    pattern_matching_kernel, THREADS_PER_BLOCK, %ld);\n", block_shmem_size);
    write_code("int nr_blocks = dev_props.multiProcessorCount * max_active_blocks_per_sm;\n");
    // write_code("int nr_blocks = 1024;\n");
    write_code("int nr_total_warps = nr_blocks * WARPS_PER_BLOCK;\n");
    if (verbose)
        write_code("printf(\"nr_blocks=%%d\\n\", nr_blocks);\n");
    write_code("\n");

    write_code("size_t size_edge = g->e_cnt * sizeof(uint32_t);\n");
    write_code("size_t size_vertex = (g->v_cnt + 1) * sizeof(uint32_t);\n");
    write_code("size_t size_tmp = VertexSet::max_intersection_size * sizeof(uint32_t) * nr_total_warps * %d;\n", schedule.get_total_prefix_num());

    write_code("uint32_t *edge_from = new uint32_t[g->e_cnt];\n");
    write_code("for (uint32_t i = 0; i < g->v_cnt; ++i)\n");
    write_code("    for (uint32_t j = g->vertex[i]; j < g->vertex[i+1]; ++j)\n");
    write_code("        edge_from[j] = i;\n\n");

    write_code("uint32_t *dev_edge, *dev_edge_from, *dev_vertex, *dev_tmp;\n");
    write_code("gpuErrchk( cudaMalloc((void**)&dev_edge, size_edge));\n");
    write_code("gpuErrchk( cudaMalloc((void**)&dev_edge_from, size_edge));\n");
    write_code("gpuErrchk( cudaMalloc((void**)&dev_vertex, size_vertex));\n");
    write_code("gpuErrchk( cudaMalloc((void**)&dev_tmp, size_tmp));\n");

    write_code("gpuErrchk( cudaMemcpy(dev_edge, g->edge, size_edge, cudaMemcpyHostToDevice));\n");
    write_code("gpuErrchk( cudaMemcpy(dev_edge_from, edge_from, size_edge, cudaMemcpyHostToDevice));\n");
    write_code("gpuErrchk( cudaMemcpy(dev_vertex, g->vertex, size_vertex, cudaMemcpyHostToDevice));\n\n");

    write_code("unsigned long long sum = 0;\n");
    write_code("unsigned cur_edge = 0;\n");
    write_code("cudaMemcpyToSymbol(dev_sum, &sum, sizeof(sum));\n");
    write_code("cudaMemcpyToSymbol(dev_cur_edge, &cur_edge, sizeof(cur_edge));\n\n");

    write_code("auto t2 = system_clock::now();\n");
    write_code("double prepare_time = 1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();\n");
    write_code("if (p_prepare_time) *p_prepare_time = prepare_time;\n");
    if (verbose)
        write_code("printf(\"prepare time: %%g seconds\\n\", prepare_time);\n");
    write_code("\n");

    write_code("auto t3 = system_clock::now();\n");
    write_code("pattern_matching_kernel<<<nr_blocks, THREADS_PER_BLOCK, %ld>>>\n", block_shmem_size);
    write_code("    (g->e_cnt, VertexSet::max_intersection_size, dev_edge_from, dev_edge, dev_vertex, dev_tmp);\n");
    write_code("gpuErrchk( cudaPeekAtLastError() );\n");
    write_code("gpuErrchk( cudaDeviceSynchronize() );\n");
    write_code("gpuErrchk( cudaMemcpyFromSymbol(&sum, dev_sum, sizeof(sum)) );\n\n");

    write_code("sum /= %d; // IEP redundancy\n\n", schedule.get_in_exclusion_optimize_redundancy());

    write_code("auto t4 = system_clock::now();\n");
    write_code("double count_time = 1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();\n");
    write_code("if (p_count_time) *p_count_time = count_time;\n");
    if (verbose) {
        write_code("printf(\"counting time: %%g seconds\\n\", count_time);\n");
        write_code("printf(\"count: %%llu\\n\", sum);\n");
    }
    write_code("\n");

    write_code("gpuErrchk(cudaFree(dev_edge));\n");
    write_code("gpuErrchk(cudaFree(dev_edge_from));\n");
    write_code("gpuErrchk(cudaFree(dev_vertex));\n");
    write_code("gpuErrchk(cudaFree(dev_tmp));\n");
    write_code("delete[] edge_from;\n");
    write_code("return sum;\n");

    indentation -= 4;
    write_code("}\n");
}

int main(int argc, char* argv[])
{
    int fd_stdout = dup(STDOUT_FILENO);
    dup2(STDERR_FILENO, STDOUT_FILENO);

    if (argc != 4) {
        printf("sample input: %s ~/data/wiki-vote.g 3 011101011", argv[0]);
        return 0;
    }
    
    Graph *g;
    DataLoader D;

    bool ok = D.fast_load(g, argv[1]);
    if (!ok) {
        printf("failed to load dataset %s\n", argv[1]);
        return 0;
    }

    int pattern_size = atoi(argv[2]);
    const char *adj_mat = argv[3];

    bool is_pattern_valid;
    Pattern p(pattern_size, adj_mat);
    Schedule_IEP schedule(p, is_pattern_valid, 1, 1, true, g->v_cnt, g->e_cnt, g->tri_cnt);

    if (!is_pattern_valid) {
        printf("invalid pattern\n");
        return 0;
    }

    dup2(fd_stdout, STDOUT_FILENO);

    emit_pattern_matching_kernel(schedule);
    emit_kernel_launch(schedule, true);

    // printf("length of vid=%ld vf=%ld vc=%ld c=%ld f=%ld ap=%ld\n",
    //     schedule.in_exclusion_optimize_vertex_id.size(),
    //     schedule.in_exclusion_optimize_vertex_flag.size(),
    //     schedule.in_exclusion_optimize_vertex_coef.size(),
    //     schedule.in_exclusion_optimize_coef.size(),
    //     schedule.in_exclusion_optimize_flag.size(),
    //     schedule.in_exclusion_optimize_ans_pos.size());

    return 0;
}
