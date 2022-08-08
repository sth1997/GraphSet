#pragma once
#include <cstdio>
#include <iostream>
#include <cerrno>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include <x86intrin.h>
#include <unistd.h>


#define SIMD_STATE 4 // 0:none, 2:scalar2x, 4:simd4x
#define SIMD_MODE 1 // 0:naive 1: filter
typedef int PackBase;
#ifdef SI64
typedef long long PackState;
#else
typedef int PackState;
#endif
const int PACK_WIDTH = sizeof(PackState) * 8;
const int PACK_SHIFT = __builtin_ctzll(PACK_WIDTH);
const int PACK_MASK = PACK_WIDTH - 1;

const size_t PARA_DEG_M128 = sizeof(__m128i) / sizeof(PackState);
const size_t PARA_DEG_M256 = sizeof(__m256i) / sizeof(PackState);

const size_t PACK_NODE_POOL_SIZE = 1024000000;

const int CACHE_LINE_SIZE = sysconf (_SC_LEVEL1_DCACHE_LINESIZE); // in byte.
struct PackNode
{
    PackBase base;
    PackState state;

    PackNode() {};
    PackNode(PackBase _b, PackState _s): base(_b), state(_s) {};
};

struct UVertex
{
    int start, deg;
    UVertex(): start(-1), deg(0) {};
    UVertex(int _s, int _d): start(_s), deg(_d) {};
};

struct DVertex
{
    int out_start, out_deg;
    int in_start, in_deg;

    DVertex(): out_start(-1), out_deg(0), in_start(-1), in_deg(0) {};
};

typedef std::pair<int, int> Edge;
typedef std::vector<std::pair<int,int>> EdgeVector;

void quit();
std::string extract_filename(const std::string full_filename);
int arg_pos(char *str, int argc, char **argv);
void align_malloc(void **memptr, size_t alignment, size_t size);
EdgeVector load_graph(const std::string path);
void save_graph(const std::string path, const EdgeVector& edge_vec);
std::vector<int> load_vertex_order(const std::string path);
void save_newid(const std::string path, std::vector<int> org2newid);
bool edge_idpair_cmp(const Edge& a, const Edge& b);

int intersect(const int *set_a, int size_a,const int *set_b, int size_b, int *set_c);
int intersect_count(const int *set_a, int size_a, const int *set_b, int size_b);

int intersect_scalar2x(const int *set_a, int size_a, const int *set_b, int size_b, int *set_c);
int intersect_scalar2x_count(int* set_a, int size_a, int* set_b, int size_b);

int intersect_simd4x(const int *set_a, int size_a, const int *set_b, int size_b, int *set_c);
int intersect_simd4x_count(int* set_a, int size_a, int* set_b, int size_b);

int intersect_filter_simd4x(const int *set_a, int size_a, const int *set_b, int size_b, int *set_c);
int intersect_filter_simd4x_count(int* set_a, int size_a, int* set_b, int size_b);

int bp_intersect(int* bases_a, PackState* states_a, int size_a,
            int* bases_b, PackState* states_b, int size_b,
            int *bases_c, PackState* states_c);
int bp_intersect_count(int* bases_a, PackState* states_a, int size_a,
            int* bases_b, PackState* states_b, int size_b);

int bp_intersect_scalar2x(int* bases_a, PackState* states_a, int size_a,
            int* bases_b, PackState* states_b, int size_b,
            int *bases_c, PackState* states_c);
int bp_intersect_scalar2x_count(int* bases_a, PackState* states_a, int size_a,
            int* bases_b, PackState* states_b, int size_b);

int bp_intersect_simd4x(int* bases_a, PackState* states_a, int size_a,
            int* bases_b, PackState* states_b, int size_b,
            int *bases_c, PackState* states_c);
int bp_intersect_simd4x_count(int* bases_a, PackState* states_a, int size_a,
            int* bases_b, PackState* states_b, int size_b);

int bp_intersect_filter_simd4x(int* bases_a, PackState* states_a, int size_a,
            int* bases_b, PackState* states_b, int size_b,
            int *bases_c, PackState* states_c);
int bp_intersect_filter_simd4x_count(int* bases_a, PackState* states_a, int size_a,
            int* bases_b, PackState* states_b, int size_b);

int merge(const int *set_a, int size_a, const int *set_b, int size_b, int *set_c);
// merge one element in-place.
int bp_merge_one(int* bases_a, PackState* states_a, int size_a,
            int v_base, PackState v_bit);

int bp_subtract_visited(int* bases_a, PackState* states_a, int size_a,
            PackState* visited, int* bases_c, PackState* states_c);
int bp_subtract_unvisited(int* bases_a, PackState* states_a, int size_a,
            PackState* visited, int* bases_c, PackState* states_c);

int bp_subtract_visited_simd4x(int* bases_a, PackState* states_a, int size_a,
            PackState* visited, int* bases_c, PackState* states_c);
int bp_subtract_unvisited_simd4x(int* bases_a, PackState* states_a, int size_a,
            PackState* visited, int* bases_c, PackState* states_c);

int subtract(const int *set_a, int size_a, const int *set_b, int size_b, int *set_c);
int bp_subtract(int* bases_a, PackState* states_a, int size_a,
            int* bases_b, PackState* states_b, int size_b,
            int *bases_c, PackState* states_c);

extern unsigned long long inter_cnt, no_match_cnt, byte_check_cnt[4], cmp_cnt, multimatch_cnt, skew_cnt, low_select_cnt;
