/**
 * 这个版本里面没有细粒度计时。有计时的在gpu_graph_with_timer.cu里面。
 * 而且计时的方式与zms版本略有区别。
 */
#include <graph.h>
#include <dataloader.h>
#include <vertex_set.h>
#include <common.h>
#include <schedule_IEP.h>
#include <motif_generator.h>

#include <cassert>
#include <cstring>
#include <cstdint>
#include <string>
#include <iostream>
#include <algorithm>
#include <cstdarg>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <sys/time.h>
#include <chrono>
#include <unistd.h>

#include "timeinterval.h"
#include "component/utils.cuh"
#include "component/gpu_schedule.cuh"
#include "component/gpu_vertex_set.cuh"
#include "component/gpu_device_context.cuh"
#include "function/pattern_matching.cuh"

int stdout_fd;
int indentation = 0;

TimeInterval allTime;
TimeInterval tmpTime;

#define gen(str) do { \
    for (int tmpi = 0; tmpi < indentation; ++tmpi) \
        printf(" "); \
    printf("%s", str); \
} while (0)


void print_statement(const char *fmt...) {
    va_list args;
    va_start(args, fmt);
    for(int _ = 0; _ < indentation; _++) putchar(' '); 
    vprintf(fmt, args);
}

void gen_build_vertex_set(const GPUSchedule* schedule, int prefix_id, int indentation, const char* input_data_str, const char* input_size_str)
{
    int father_id = schedule->get_father_prefix_id(prefix_id);
    if (father_id == -1)
    {
        print_statement("if (threadIdx.x % THREADS_PER_WARP == 0)\n");
        print_statement("    vertex_set[%d].init(%s, %s);\n", prefix_id, input_size_str, input_data_str);
        print_statement("__threadfence_block();\n");
    }
    else
    {
        bool only_need_size = schedule->only_need_size[prefix_id];
        static bool first_time = true;
        if (first_time)
            print_statement("GPUVertexSet* tmp_vset;\n");
        first_time = false;
        if(only_need_size) {
            print_statement("{\n");
            print_statement("tmp_vset = &vertex_set[%d];\n", prefix_id);
            print_statement("if (threadIdx.x % THREADS_PER_WARP == 0)\n");
            print_statement("    tmp_vset->init(%s, %s);\n", input_size_str, input_data_str);
            print_statement("__threadfence_block();\n");
            print_statement("if (%s > vertex_set[%d].get_size())\n", input_size_str, father_id);
            print_statement("    tmp_vset->size -= unordered_subtraction_size(*tmp_vset, vertex_set[%d], -1);\n", father_id);
            print_statement("else\n");
            print_statement("    tmp_vset->size = vertex_set[%d].get_size() - unordered_subtraction_size(vertex_set[%d], *tmp_vset, -1);\n", father_id, father_id);
            print_statement("}\n");
        }
        else {
            print_statement("intersection2(vertex_set[%d].get_data_ptr(), vertex_set[%d].get_data_ptr(), %s, vertex_set[%d].get_size(), %s, &vertex_set[%d].size);\n", prefix_id, father_id, input_data_str, father_id, input_size_str, prefix_id);
        }
    }
}


 void gen_GPU_pattern_matching_func(const PatternMatchingDeviceContext * context)

{
    GPUSchedule *schedule = context->dev_schedule;

    // print include statement first

    // print_statement("#include \"component/gpu_device_context.cuh\"\n");
    // print_statement("#include \"component/gpu_schedule.cuh\"\n");
    // print_statement("#include \"component/gpu_vertex_set.cuh\"\n");
    // print_statement("#include \"component/utils.cuh\"\n");

    // print_statement("#include \"function/pattern_matching.cuh\"\n");


    //WORK SPACE BEGIN
    //如果图也能确定的话，edge_num也可以确定
    print_statement("__global__ void gpu_pattern_matching_generated(e_index_t edge_num, uint32_t buffer_size, PatternMatchingDeviceContext *context) {\n");
    indentation += 4;
    print_statement("__shared__ e_index_t block_edge_idx[WARPS_PER_BLOCK];\n");
    print_statement("extern __shared__ GPUVertexSet block_vertex_set[];\n");

    int num_prefixes = schedule->get_total_prefix_num();
    int num_vertex_sets_per_warp = num_prefixes + 2;

    print_statement("GPUSchedule *schedule = context->dev_schedule;\n");
    print_statement("uint32_t *tmp = context->dev_tmp;\n");
    print_statement("uint32_t *edge = (uint32_t *)context->dev_edge;\n");
    print_statement("e_index_t *vertex = context->dev_vertex;\n");
    print_statement("uint32_t *edge_from = (uint32_t *)context->dev_edge_from;\n");

    print_statement("int wid = threadIdx.x / THREADS_PER_WARP, lid = threadIdx.x % THREADS_PER_WARP, global_wid = blockIdx.x * WARPS_PER_BLOCK + wid;\n");
    print_statement("e_index_t &edge_idx = block_edge_idx[wid];\n");
    print_statement("GPUVertexSet *vertex_set = block_vertex_set + wid * %d;\n", num_vertex_sets_per_warp);
    
    print_statement("if (lid == 0) {\n");
    indentation += 4;
    print_statement("edge_idx = 0;\n");
    print_statement("uint32_t offset = buffer_size * global_wid * %d;\n",num_vertex_sets_per_warp);
    
    print_statement("for (int i = 0; i < %d; ++i) {\n", num_vertex_sets_per_warp);
    indentation += 4;
    print_statement("vertex_set[i].set_data_ptr(tmp + offset);\n");
    print_statement("offset += buffer_size;\n");
    indentation -= 4;
    print_statement("}\n");
    indentation -= 4;
    print_statement("}\n");


    print_statement("GPUVertexSet& subtraction_set = vertex_set[%d];\n", num_prefixes);
    //print_statement("GPUVertexSet& tmp_set = vertex_set[%d];\n", num_prefixes + 1);

    print_statement("__threadfence_block();\n");

    print_statement("uint32_t v0, v1;\n");
    print_statement("e_index_t l, r;\n");

    print_statement("unsigned long long sum = 0;\n");

    print_statement("while (true) {\n");
    indentation += 4;
    print_statement("if (lid == 0) {\n");
    indentation += 4;
    print_statement("edge_idx = atomicAdd(context->dev_cur_edge, 1);\n");
    print_statement("unsigned int i = edge_idx;\n");
    print_statement("if (i < edge_num) {\n");
    indentation += 4;
    print_statement("subtraction_set.init();\n");
    print_statement("subtraction_set.push_back(edge_from[i]);\n");
    print_statement("subtraction_set.push_back(edge[i]);\n");
    indentation -= 4;
    print_statement("}\n");
    indentation -= 4;
    print_statement("}\n");

    print_statement("__threadfence_block();\n");

    print_statement("e_index_t i = edge_idx;\n");
    print_statement("if(i >= edge_num) break;\n");
    
    print_statement("v0 = edge_from[i];\n");
    print_statement("v1 = edge[i];\n");

    //print_statement("bool is_zero = false;\n");
    print_statement("get_edge_index(v0, l, r);\n");
    for (int prefix_id = schedule->get_last(0); prefix_id != -1; prefix_id = schedule->get_next(prefix_id)) {
        gen_build_vertex_set(schedule, prefix_id, indentation, "&edge[l]", "r - l");
    }
    
    if (schedule->get_restrict_last(1) != -1) {
        print_statement("if(v0 <= v1) continue;\n");
    } 

    print_statement("get_edge_index(v1, l, r);\n");
    for (int prefix_id = schedule->get_last(1); prefix_id != -1; prefix_id = schedule->get_next(prefix_id))
    {
        //print_statement("vertex_set[%d].build_vertex_set(%d, vertex_set, &edge[l], r - l);\n", prefix_id, schedule->get_father_prefix_id(prefix_id));
        gen_build_vertex_set(schedule, prefix_id, indentation, "&edge[l]", "r - l");
        if (prefix_id < schedule->get_basic_prefix_num())
            print_statement("if (vertex_set[%d].get_size() == 0) continue;\n", prefix_id); //因为代码生成后没有外层的for循环了，所以不需要先break再continue了
    }
    
    print_statement("extern __shared__ char ans_array[];\n");
    print_statement("int* ans = ((int*) (ans_array + %d)) + %d * (threadIdx.x / THREADS_PER_WARP);\n", schedule->ans_array_offset, schedule->in_exclusion_optimize_vertex_id_size);
    
    for(int depth = 2; depth < schedule->get_size() - schedule->get_in_exclusion_optimize_num();  ++depth) {
        int loop_set_prefix_id = schedule->get_loop_set_prefix_id(depth);
        print_statement("int loop_size_depth%d = vertex_set[%d].get_size();\n", depth, loop_set_prefix_id);
        print_statement("if( loop_size_depth%d <= 0) continue;\n", depth);

        print_statement("uint32_t* loop_data_ptr_depth%d = vertex_set[%d].get_data_ptr();\n",depth, loop_set_prefix_id);
        
        bool has_min_vertex = schedule->get_restrict_last(depth) != -1;
        // if(has_min_vertex) {
            print_statement("uint32_t min_vertex_depth%d = 0xffffffff;\n", depth);
        // }

        for (int i = schedule->get_restrict_last(depth); i != -1; i = schedule->get_restrict_next(i)) {
            print_statement("if(min_vertex_depth%d > subtraction_set.get_data(%d)) min_vertex_depth%d = subtraction_set.get_data(%d);\n", depth, schedule->get_restrict_index(i), depth, schedule->get_restrict_index(i));
        }

        if (depth == schedule->get_size() - 1 && schedule->get_in_exclusion_optimize_num() == 0) {
            print_statement("int size_after_restrict = lower_bound(loop_data_ptr_depth%d, loop_size_depth%d, min_vertex_depth%d);\n", depth, depth, depth);
            print_statement("sum += unordered_subtraction_size(vertex_set[%d], subtraction_set, size_after_restrict);\n", loop_set_prefix_id);
            break;
        }

        print_statement("for(int i_depth%d = 0; i_depth%d < loop_size_depth%d; ++i_depth%d) {\n", depth, depth, depth, depth);
        indentation += 4;
        print_statement("uint32_t v_depth%d = loop_data_ptr_depth%d[i_depth%d];\n", depth, depth, depth);
        if(has_min_vertex) {
            print_statement("if (min_vertex_depth%d <= v_depth%d) break;\n", depth, depth);
        }
        print_statement("if(subtraction_set.has_data(v_depth%d)) continue;\n", depth);
        print_statement("unsigned int l_depth%d, r_depth%d;\n", depth, depth);
        print_statement("get_edge_index(v_depth%d, l_depth%d, r_depth%d);\n", depth, depth, depth);

        for (int prefix_id = schedule->get_last(depth); prefix_id != -1; prefix_id = schedule->get_next(prefix_id)) {
            char data_str[64] = "&edge[l_depth?]";
            data_str[13] = depth + '0';
            char size_str[64] = "r_depth? - l_depth?";
            size_str[7] = depth + '0';
            size_str[18] = depth + '0';
            //sprint_statement(data_str, "&edge[l_depth%d]", depth);
            //sprint_statement(size_str, "r_depth%d - l_depth%d", depth, depth);
            gen_build_vertex_set(schedule, prefix_id, indentation, data_str, size_str);
            if (prefix_id < schedule->get_basic_prefix_num()) //在general版本中没有这个判断，是因为general加了这个判断会更慢（不如直接判断break_size），但在这个版本加了这个判断可能会省去之后的break_size判断
                print_statement("if (vertex_set[%d].get_size() == %d) continue;\n", prefix_id, schedule->get_break_size(prefix_id));
        }
        print_statement("if (threadIdx.x % THREADS_PER_WARP == 0) subtraction_set.push_back(v_depth%d);\n", depth);
        print_statement("__threadfence_block();\n");

    }
    
    if (schedule->get_in_exclusion_optimize_num() != 0)
    {
        for(int i = 0; i < schedule->in_exclusion_optimize_vertex_id_size; ++i) {
            if(schedule->in_exclusion_optimize_vertex_flag[i]) {
                print_statement("ans[%d] = vertex_set[%d].get_size() - %d;\n", i, schedule->in_exclusion_optimize_vertex_id[i], schedule->in_exclusion_optimize_vertex_coef[i]);
            }
            else {
                print_statement("ans[%d] = unordered_subtraction_size(vertex_set[%d], subtraction_set);\n", i, schedule->in_exclusion_optimize_vertex_id[i]);
            }
        }
        int last_pos = -1;
        print_statement("long long val;\n");
        for(int pos = 0; pos < schedule->in_exclusion_optimize_array_size; ++pos) {
            if(pos == last_pos + 1) {
                print_statement("val = ans[%d];\n", schedule->in_exclusion_optimize_ans_pos[pos]);
            }
            else {
                print_statement("val = val * ans[%d];\n", schedule->in_exclusion_optimize_ans_pos[pos]);
            }
            if(schedule->in_exclusion_optimize_flag[pos]) {
                last_pos = pos;
                print_statement("sum += val * %d;\n", schedule->in_exclusion_optimize_coef[pos]);
            }
        }
    }

    int last_depth;
    if (schedule->get_in_exclusion_optimize_num() != 0)
        last_depth = schedule->get_size() - schedule->get_in_exclusion_optimize_num() - 1;
    else
        last_depth = schedule->get_size() - 2;
    for(int depth = last_depth; depth >= 2; --depth) {
        print_statement("if (threadIdx.x % THREADS_PER_WARP == 0) subtraction_set.pop_back();\n");
        print_statement("__threadfence_block();\n");
        indentation -= 4;
        print_statement("}\n");
    }
    indentation -= 4;
    print_statement("}\n");
    print_statement("if (lid == 0) atomicAdd(context->dev_sum, sum);\n");
    indentation -= 4;
    print_statement("}\n");
}

void pattern_matching(Graph *g, const Schedule_IEP& schedule_iep) {
    tmpTime.check();

    PatternMatchingDeviceContext * context;
    gpuErrchk(cudaMallocManaged((void **)&context, sizeof(PatternMatchingDeviceContext)));
    context->init(g, schedule_iep);

    //tmpTime.print("Prepare time cost");
    tmpTime.check();

    fflush(stdout);
    dup2(stdout_fd, STDOUT_FILENO); //之前重定向到文件主要是为了把一些初始化的输出重定向，现在重定向回控制台

    gen_GPU_pattern_matching_func(context);
    fflush(stdout);

    context->destroy();
    gpuErrchk(cudaFree(context));
}

int main(int argc,char *argv[]) {
    stdout_fd = dup(STDOUT_FILENO);
    freopen("tmp", "w", stdout);
    Graph *g;
    DataLoader D;

    using std::chrono::system_clock;
    auto t1 = system_clock::now();

    bool ok;
    
    ok = D.fast_load(g, argv[1]);
    
    if (!ok) {
        printf("data load failure :-(\n");
        return 1;
    }

    auto t2 = system_clock::now();
    auto load_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    // printf("Load data success! time: %g seconds\n", load_time.count() / 1.0e6);
    // fflush(stdout);

    int pattern_size = atoi(argv[2]);
    const char* pattern_str= argv[3];

    Pattern p(pattern_size, pattern_str);
    //printf("pattern = \n");
    //p.print();
    //printf("max intersection size %d\n", VertexSet::max_intersection_size);
    bool is_pattern_valid;
    bool use_in_exclusion_optimize = true;
    Schedule_IEP schedule_iep(p, is_pattern_valid, 1, 1, use_in_exclusion_optimize, g->v_cnt, g->e_cnt, g->tri_cnt);

    if (!is_pattern_valid) {
        printf("pattern is invalid!\n");
        return 1;
    }

    pattern_matching(g, schedule_iep);

    //allTime.print("Total time cost");

    return 0;
}
