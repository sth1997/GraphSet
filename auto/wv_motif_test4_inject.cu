__global__ void gpu_pattern_matching(uint32_t edge_num, uint32_t buffer_size, uint32_t *edge_from, uint32_t *edge, uint32_t *vertex, uint32_t *tmp, const GPUSchedule* schedule) {
__shared__ unsigned int block_edge_idx[WARPS_PER_BLOCK];
extern __shared__ GPUVertexSet block_vertex_set[];
int wid = threadIdx.x / THREADS_PER_WARP;
int lid = threadIdx.x % THREADS_PER_WARP;
int global_wid = blockIdx.x * WARPS_PER_BLOCK + wid;
unsigned int &edge_idx = block_edge_idx[wid];
GPUVertexSet *vertex_set = block_vertex_set + wid * 4;
if (lid == 0) {
edge_idx = 0;
uint32_t offset = buffer_size * global_wid * 4;
for (int i = 0; i < 4; ++i) {
vertex_set[i].set_data_ptr(tmp + offset);
offset += buffer_size;
}
}
GPUVertexSet& subtraction_set = vertex_set[2];
__threadfence_block();
uint32_t v0, v1;
uint32_t l, r;
unsigned long long sum = 0;
while (true) {
if (lid == 0) {
edge_idx = atomicAdd(&dev_cur_edge, 1);
unsigned int i = edge_idx;
if (i < edge_num) {
subtraction_set.init();
subtraction_set.push_back(edge_from[i]);
subtraction_set.push_back(edge[i]);
}
}
__threadfence_block();
unsigned int i = edge_idx;
if(i >= edge_num) break;
v0 = edge_from[i];
v1 = edge[i];
get_edge_index(v0, l, r);
if (threadIdx.x % THREADS_PER_WARP == 0)
    vertex_set[0].init(r - l, &edge[l]);
__threadfence_block();
if(v0 <= v1) continue;
get_edge_index(v1, l, r);
GPUVertexSet* tmp_vset;
{
tmp_vset = &vertex_set[1];
if (threadIdx.x % THREADS_PER_WARP == 0)
    tmp_vset->init(r - l, &edge[l]);
__threadfence_block();
if (r - l > vertex_set[0].get_size())
    tmp_vset->size -= unordered_subtraction_size(*tmp_vset, vertex_set[0], -1);
else
    tmp_vset->size = vertex_set[0].get_size() - unordered_subtraction_size(vertex_set[0], *tmp_vset, -1);
}
if (vertex_set[1].get_size() == 0) continue;
extern __shared__ char ans_array[];
int* ans = ((int*) (ans_array + 512)) + 1 * (threadIdx.x / THREADS_PER_WARP);
ans[0] = vertex_set[1].get_size() - 0;
long long val;
val = ans[0];
val = val * ans[0];
sum += val * 1;
val = ans[0];
sum += val * -1;
}
if (lid == 0) atomicAdd(&dev_sum, sum);
}
