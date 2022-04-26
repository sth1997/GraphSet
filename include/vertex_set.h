#pragma once
#include "schedule.h"
#include <bitset>

void long_add(long long &low, long long &high, int num);


struct Bitmap{
    unsigned long long *s;
    Bitmap(int size);
    ~Bitmap();
    inline void set_bit(int pos);
    inline bool read_bit(int pos);
    inline void set_0();
};

class VertexSet
{
public:
    VertexSet();
    // allocate new memory according to max_intersection_size
    void init();
    // use memory from Graph, do not allocate new memory
    void init(int input_size, int* input_data);
    void copy(int input_size, int* input_data);
    ~VertexSet();
    void intersection(const VertexSet& set0, const VertexSet& set1, int min_vertex = -1, bool clique = false);
    void intersection(const VertexSet& set0, int *input_data, int input_size);
    void intersection_with(const VertexSet& set1);
    void intersection_bs(const VertexSet &set0, int * input_data, int input_size);
    void build_bitmap();
    //set1 is unordered
    static int unorderd_subtraction_size(const VertexSet& set0, const VertexSet& set1, int size_after_restrict = -1);
    void insert_ans_sort(int val);
    inline int get_size() const { return size;}
    inline int get_data(int i) const { return data[i];}
    inline const int* get_data_ptr() const { return data;}
    inline int* get_data_ptr() { return data;}
    inline void push_back(int val) { data[size++] = val;}
    inline void pop_back() { --size;}
    inline int get_last() const { return data[size - 1];}
    bool has_data(int val);
    static int max_intersection_size;
    void build_vertex_set(const Schedule& schedule, const VertexSet* vertex_set, int* input_data, int input_size, int prefix_id, int min_vertex = -1, bool clique = false);
private:
    int* data;
    int size;
    bool allocate;
    Bitmap *bs;
};
