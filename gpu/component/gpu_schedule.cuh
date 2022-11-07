#pragma once

#include <cstdint>
#include "utils.cuh"

class GPUSchedule {
public:
    /*
    __host__ void transform_in_exclusion_optimize_group_val(const Schedule& schedule)
    {
        // @todo 注意当容斥优化无法使用时，内存分配会失败。需要修正 
        int in_exclusion_optimize_num = schedule.get_in_exclusion_optimize_num();
        gpuErrchk( cudaMallocManaged((void**)&in_exclusion_optimize_val, sizeof(int) * schedule.in_exclusion_optimize_val.size()));
        for (auto val : schedule.in_exclusion_optimize_val)
            in_exclusion_optimize_val[in_exclusion_optimize_val_size++] = val;
        in_exclusion_optimize_val_size = schedule.in_exclusion_optimize_val.size();
        
        //这部分有太多重复访存操作了（比如循环中的.data[i].data[j]，直接用一个tmp指针就行了），之后考虑优化掉（不过感觉O3会帮忙自动优化的）
        in_exclusion_optimize_group.size = schedule.in_exclusion_optimize_group.size();
        gpuErrchk( cudaMallocManaged((void**)&in_exclusion_optimize_group.data, sizeof(GPUGroupDim1) * in_exclusion_optimize_group.size));
        for (int i = 0; i < schedule.in_exclusion_optimize_group.size(); ++i)
        {
            in_exclusion_optimize_group.data[i].size = schedule.in_exclusion_optimize_group[i].size();
            gpuErrchk( cudaMallocManaged((void**)&in_exclusion_optimize_group.data[i].data, sizeof(GPUGroupDim2) * in_exclusion_optimize_group.data[i].size));
            for (int j = 0; j < schedule.in_exclusion_optimize_group[i].size(); ++j)
            {
                in_exclusion_optimize_group.data[i].data[j].size = schedule.in_exclusion_optimize_group[i][j].size();
                gpuErrchk( cudaMallocManaged((void**)&in_exclusion_optimize_group.data[i].data[j].data, sizeof(int) * in_exclusion_optimize_group.data[i].data[j].size));
                for (int k = 0; k < schedule.in_exclusion_optimize_group[i][j].size(); ++k)
                    in_exclusion_optimize_group.data[i].data[j].data[k] = schedule.in_exclusion_optimize_group[i][j][k];
            }
        }
    }
    */

    inline __device__ int get_total_prefix_num() const { return total_prefix_num;}
    inline __device__ int get_basic_prefix_num() const { return basic_prefix_num;}
    inline __device__ int get_father_prefix_id(int prefix_id) const { return father_prefix_id[prefix_id];}
    inline __device__ int get_loop_set_prefix_id(int loop) const { return loop_set_prefix_id[loop];}
    inline __device__ int get_size() const { return size;}
    inline __device__ int get_last(int i) const { return last[i];}
    inline __device__ int get_next(int i) const { return next[i];}
    inline __device__ int get_prefix_target(int i) const {return prefix_target[i];}
    inline __device__ int get_break_size(int i) const { return break_size[i];}
    inline __device__ int get_in_exclusion_optimize_num() const { return in_exclusion_optimize_num;}
    inline __device__ int get_total_restrict_num() const { return total_restrict_num;}
    inline __device__ int get_restrict_last(int i) const { return restrict_last[i];}
    inline __device__ int get_restrict_next(int i) const { return restrict_next[i];}
    inline __device__ int get_restrict_index(int i) const { return restrict_index[i];}
    //inline __device__ int get_k_val() const { return k_val;} // see below (the k_val's definition line) before using this function

    int* adj_mat;
    int* father_prefix_id;
    int* last;
    int* next;
    int* break_size;
    int* loop_set_prefix_id;
    int* prefix_target;
    int* restrict_last;
    int* restrict_next;
    int* restrict_index;
    bool* only_need_size;
    //int* in_exclusion_optimize_val;
    //GPUGroupDim0 in_exclusion_optimize_group;
    //int in_exclusion_optimize_val_size;
    int size;
    int total_prefix_num;
    int basic_prefix_num;
    int total_restrict_num;
    int in_exclusion_optimize_num;
    //int k_val;

    int in_exclusion_optimize_vertex_id_size;
    int* in_exclusion_optimize_vertex_id;
    bool* in_exclusion_optimize_vertex_flag;
    int* in_exclusion_optimize_vertex_coef;
    
    int in_exclusion_optimize_array_size;
    int* in_exclusion_optimize_coef;
    bool* in_exclusion_optimize_flag;
    int* in_exclusion_optimize_ans_pos;

    uint32_t ans_array_offset;

    bool is_vertex_induced;
    uint32_t p_label_offset;
    int max_edge;

    void create_from_schedule(const Schedule_IEP& schedule) {
        int schedule_size = schedule.get_size();
        int max_prefix_num = schedule_size * (schedule_size - 1) / 2 + 1;

        // for only_need_size
        auto only_need_size = new bool[max_prefix_num];
        for (int i = 0; i < max_prefix_num; ++i)
            only_need_size[i] = schedule.get_prefix_only_need_size(i);
        
        gpuErrchk( cudaMallocManaged((void**)&this->only_need_size, sizeof(bool) * max_prefix_num));
        gpuErrchk( cudaMemcpy(this->only_need_size, only_need_size, sizeof(bool) * max_prefix_num, cudaMemcpyHostToDevice));

        // for in-exclusion
        this->in_exclusion_optimize_array_size = in_exclusion_optimize_array_size;
        this->in_exclusion_optimize_vertex_id_size = in_exclusion_optimize_vertex_id_size;
        this->in_exclusion_optimize_num = schedule.get_in_exclusion_optimize_num();
        
        int in_exclusion_optimize_vertex_id_size = schedule.in_exclusion_optimize_vertex_id.size();
        int in_exclusion_optimize_array_size = schedule.in_exclusion_optimize_coef.size();

        auto in_exclusion_optimize_vertex_id = &(schedule.in_exclusion_optimize_vertex_id[0]);
        auto in_exclusion_optimize_vertex_coef = &(schedule.in_exclusion_optimize_vertex_coef[0]);
        auto in_exclusion_optimize_vertex_flag = new bool[in_exclusion_optimize_vertex_id_size]; 

        auto in_exclusion_optimize_coef = &(schedule.in_exclusion_optimize_coef[0]);
        auto in_exclusion_optimize_ans_pos = &(schedule.in_exclusion_optimize_ans_pos[0]);
        auto in_exclusion_optimize_flag = new bool[in_exclusion_optimize_array_size];

        for (int i = 0; i < in_exclusion_optimize_vertex_id_size; ++i)
            in_exclusion_optimize_vertex_flag[i] = schedule.in_exclusion_optimize_vertex_flag[i];
        
        for (int i = 0; i < in_exclusion_optimize_array_size; ++i)
            in_exclusion_optimize_flag[i] = schedule.in_exclusion_optimize_flag[i];

        gpuErrchk( cudaMallocManaged((void**)&this->in_exclusion_optimize_vertex_id, sizeof(int) * in_exclusion_optimize_vertex_id_size));
        gpuErrchk( cudaMemcpy(this->in_exclusion_optimize_vertex_id, in_exclusion_optimize_vertex_id, sizeof(int) * in_exclusion_optimize_vertex_id_size, cudaMemcpyHostToDevice));
        
        gpuErrchk( cudaMallocManaged((void**)&this->in_exclusion_optimize_vertex_flag, sizeof(bool) * in_exclusion_optimize_vertex_id_size));
        gpuErrchk( cudaMemcpy(this->in_exclusion_optimize_vertex_flag, in_exclusion_optimize_vertex_flag, sizeof(bool) * in_exclusion_optimize_vertex_id_size, cudaMemcpyHostToDevice));
        
        gpuErrchk( cudaMallocManaged((void**)&this->in_exclusion_optimize_vertex_coef, sizeof(int) * in_exclusion_optimize_vertex_id_size));
        gpuErrchk( cudaMemcpy(this->in_exclusion_optimize_vertex_coef, in_exclusion_optimize_vertex_coef, sizeof(int) * in_exclusion_optimize_vertex_id_size, cudaMemcpyHostToDevice));

        gpuErrchk( cudaMallocManaged((void**)&this->in_exclusion_optimize_coef, sizeof(int) * in_exclusion_optimize_array_size));
        gpuErrchk( cudaMemcpy(this->in_exclusion_optimize_coef, in_exclusion_optimize_coef, sizeof(int) * in_exclusion_optimize_array_size, cudaMemcpyHostToDevice));

        gpuErrchk( cudaMallocManaged((void**)&this->in_exclusion_optimize_flag, sizeof(bool) * in_exclusion_optimize_array_size));
        gpuErrchk( cudaMemcpy(this->in_exclusion_optimize_flag, in_exclusion_optimize_flag, sizeof(bool) * in_exclusion_optimize_array_size, cudaMemcpyHostToDevice));
        
        gpuErrchk( cudaMallocManaged((void**)&this->in_exclusion_optimize_ans_pos, sizeof(int) * in_exclusion_optimize_array_size));
        gpuErrchk( cudaMemcpy(this->in_exclusion_optimize_ans_pos, in_exclusion_optimize_ans_pos, sizeof(int) * in_exclusion_optimize_array_size, cudaMemcpyHostToDevice));

            

        gpuErrchk( cudaMallocManaged((void**)&this->father_prefix_id, sizeof(int) * max_prefix_num));
        gpuErrchk( cudaMemcpy(this->father_prefix_id, schedule.get_father_prefix_id_ptr(), sizeof(int) * max_prefix_num, cudaMemcpyHostToDevice));

        gpuErrchk( cudaMallocManaged((void**)&this->last, sizeof(int) * schedule_size));
        gpuErrchk( cudaMemcpy(this->last, schedule.get_last_ptr(), sizeof(int) * schedule_size, cudaMemcpyHostToDevice));

        gpuErrchk( cudaMallocManaged((void**)&this->next, sizeof(int) * max_prefix_num));
        gpuErrchk( cudaMemcpy(this->next, schedule.get_next_ptr(), sizeof(int) * max_prefix_num, cudaMemcpyHostToDevice));

        gpuErrchk( cudaMallocManaged((void**)&this->loop_set_prefix_id, sizeof(int) * schedule_size));
        gpuErrchk( cudaMemcpy(this->loop_set_prefix_id, schedule.get_loop_set_prefix_id_ptr(), sizeof(int) * schedule_size, cudaMemcpyHostToDevice));

        gpuErrchk( cudaMallocManaged((void**)&this->prefix_target, sizeof(int) * max_prefix_num));
        gpuErrchk( cudaMemcpy(this->prefix_target, schedule.get_prefix_target_ptr(), sizeof(int) * max_prefix_num, cudaMemcpyHostToDevice));


        gpuErrchk( cudaMallocManaged((void**)&this->break_size, sizeof(int) * max_prefix_num));
        gpuErrchk( cudaMemcpy(this->break_size, schedule.get_break_size_ptr(), sizeof(int) * max_prefix_num, cudaMemcpyHostToDevice));

        // for restriction
        this->total_restrict_num = schedule.get_total_restrict_num();

        gpuErrchk( cudaMallocManaged((void**)&this->restrict_last, sizeof(int) * schedule_size));
        gpuErrchk( cudaMemcpy(this->restrict_last, schedule.get_restrict_last_ptr(), sizeof(int) * schedule_size, cudaMemcpyHostToDevice));
        
        gpuErrchk( cudaMallocManaged((void**)&this->restrict_next, sizeof(int) * max_prefix_num));
        gpuErrchk( cudaMemcpy(this->restrict_next, schedule.get_restrict_next_ptr(), sizeof(int) * max_prefix_num, cudaMemcpyHostToDevice));
        
        gpuErrchk( cudaMallocManaged((void**)&this->restrict_index, sizeof(int) * max_prefix_num));
        gpuErrchk( cudaMemcpy(this->restrict_index, schedule.get_restrict_index_ptr(), sizeof(int) * max_prefix_num, cudaMemcpyHostToDevice));

        gpuErrchk( cudaMallocManaged((void**)&this->adj_mat, sizeof(int) * schedule_size * schedule_size));
        gpuErrchk( cudaMemcpy(this->adj_mat, schedule.get_adj_mat_ptr(), sizeof(int) * schedule_size * schedule_size, cudaMemcpyHostToDevice));

        this->size = schedule.get_size();
        this->total_prefix_num = schedule.get_total_prefix_num();
        this->basic_prefix_num = schedule.get_basic_prefix_num();

        delete[] only_need_size;
        delete[] in_exclusion_optimize_vertex_flag;
        delete[] in_exclusion_optimize_flag;
    
    }

};