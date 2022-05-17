#include "../include/vertex_set.h"
#include <algorithm>

int VertexSet::max_intersection_size = -1;

VertexSet::VertexSet()
:data(nullptr), size(0), allocate(false)
{}

void VertexSet::init()
{
    if (allocate == true && data != nullptr)
        size = 0; // do not reallocate
    else
    {
        size = 0;
        allocate = true;
        data = new int[max_intersection_size * 2];
    }
}

void VertexSet::init(int input_size, int* input_data)
{
    if (allocate == true && data != nullptr)
        delete[] data;
    size = input_size;
    data = input_data;
    allocate = false;
}

void VertexSet::init_bs(Bitmap* bs, int input_size, int* input_data)
{
    // assert(false);
    if (allocate == true && data != nullptr)
        delete[] data;
    size = input_size;
    data = input_data;
    for(int i = 0; i < size; i++) bs->inc(data[i]);
    allocate = false;
}


void VertexSet::intersection_bs(const VertexSet& set0, Bitmap *bs, int *input_data, int input_size, int depth) {

    const int *set0_data = set0.get_data_ptr(), *set1_data = input_data;
    int size0 = set0.get_size(), size1 = input_size;
    int i = 0, j = 0, data0, data1;
    size = 0;
    for(int i = 0; i < size1; i++) {
        if(bs->read(set1_data[i]) == depth) {
            push_back(set1_data[i]);
            bs->inc(set1_data[i]);
        }
    }
}

void VertexSet::build_vertex_set_bs(const Schedule_IEP& schedule, const VertexSet* vertex_set, Bitmap *bs, int* input_data, int input_size, int prefix_id, int depth)
{
    int father_id = schedule.get_father_prefix_id(prefix_id);
    if (father_id == -1)
        init_bs(bs, input_size, input_data);
    else
    {
        init();
        intersection_bs(vertex_set[father_id], bs, input_data, input_size, depth);
    }
}

void VertexSet::copy(int input_size, int* input_data)
{
    init();
    size = input_size;
    for(int i = 0; i < input_size; ++i) data[i] = input_data[i];
}

VertexSet::~VertexSet()
{
    if (allocate== true && data != nullptr)
        delete[] data;
}

void VertexSet::intersection(const VertexSet& set0, const VertexSet& set1, int min_vertex, bool clique)
{
    int i = 0;
    int j = 0;
    int size0 = set0.get_size();
    int size1 = set1.get_size();

//    long_add(intersection_times_low, intersection_times_high, size0);
//    long_add(intersection_times_low, intersection_times_high, size0);

    // TODO : Try more kinds of calculation.
    // Like
    // while (true)
    //     ..., if (++i == size0) break;
    //     ..., if (++j == size1) break;
    //     ......
    // Maybe we can also use binary search if one set is very small and another is large.
    if (clique == true)
        if (set0.get_data(0) >= min_vertex || set1.get_data(0) >= min_vertex)
            return;
    int data0 = set0.get_data(0);
    int data1 = set1.get_data(0);
    if (clique)
        // TODO : Try more kinds of calculation.
        // For example, we can use binary search find the last element which is smaller than min_vertex, and set its index as loop_size.
        while (i < size0 && j < size1)
        {
            if (data0 < data1)
            {
                if ((data0 = set0.get_data(++i)) >= min_vertex)
                    break;
            }
            else if (data0 > data1)
            {
                if ((data1 = set1.get_data(++j)) >= min_vertex)
                    break;
            }
            else
            {
                push_back(data0);
                if ((data0 = set0.get_data(++i)) >= min_vertex)
                    break;
                if ((data1 = set1.get_data(++j)) >= min_vertex)
                    break;
            }
        }
    else
        while (i < size0 && j < size1)
        {
            data0 = set0.get_data(i);
            data1 = set1.get_data(j);
            if (data0 < data1)
                ++i;
            else if (data0 > data1)
                ++j;
            else
            {
                push_back(data0);
                ++i;
                ++j;
            }
        }
/*    #pragma omp critical 
    {
        long_add(intersection_times_low, intersection_times_high, i);
        long_add(intersection_times_low, intersection_times_high, j);
    }*/
}

void VertexSet::intersection_with(const VertexSet& set1) {
    const VertexSet& set0 = *this;
    int i = 0;
    int j = 0;
    int size0 = set0.get_size();
    int size1 = set1.get_size();

//    long_add(intersection_times_low, intersection_times_high, size0);
//    long_add(intersection_times_low, intersection_times_high, size1);

    // TODO : Try more kinds of calculation.
    // Like
    // while (true)
    //     ..., if (++i == size0) break;
    //     ..., if (++j == size1) break;
    //     ......
    // Maybe we can also use binary search if one set is very small and another is large.
    int data0 = set0.get_data(0);
    int data1 = set1.get_data(0);
    size = 0;
    while (i < size0 && j < size1)
    {
        data0 = set0.get_data(i);
        data1 = set1.get_data(j);
        if (data0 < data1)
            ++i;
        else if (data0 > data1)
            ++j;
        else
        {
            push_back(data0);
            ++i;
            ++j;
        }
    }
/*    #pragma omp critical 
    {
        long_add(intersection_times_low, intersection_times_high, i);
        long_add(intersection_times_low, intersection_times_high, j);
    }*/
}

void VertexSet::build_vertex_set(const Schedule_IEP& schedule, const VertexSet* vertex_set, int* input_data, int input_size, int prefix_id, int min_vertex, bool clique)
{
    int father_id = schedule.get_father_prefix_id(prefix_id);
    if (father_id == -1)
        init(input_size, input_data);
    else
    {
        init();
        VertexSet tmp_vset;
        tmp_vset.init(input_size, input_data);
        intersection(vertex_set[father_id], tmp_vset, min_vertex, clique);
    }
}

void VertexSet::build_vertex_set_bs_only_size(const Schedule_IEP& schedule, const VertexSet* vertex_set, Bitmap *bs, int* input_data, int input_size, int prefix_id, int depth)
{
    int father_id = schedule.get_father_prefix_id(prefix_id);
    // assert (father_id != -1);
    size = 0;
    VertexSet tmp_vset;
    tmp_vset.init(input_size, input_data);
    intersection_bs_only_size(vertex_set[father_id], bs, tmp_vset, depth);
}

void VertexSet::intersection_bs_only_size(const VertexSet& set0, Bitmap *bs, const VertexSet& set1, int depth) {
    const int *set0_data = set0.get_data_ptr(), *set1_data = set1.get_data_ptr();
    int size0 = set0.get_size(), size1 = set1.get_size();
    int i = 0, j = 0, data0, data1;
    size = 0;
    for(int i = 0; i < size1;i++){
        if(bs->read(set1_data[i]) == depth) {
            size++;
        }
    }

}

void VertexSet::insert_ans_sort(int val)
{
    int i;
    for (i = size - 1; i >= 0; --i)
        if (data[i] >= val)
            data[i + 1] = data[i];
        else
        {
            data[i + 1] = val;
            break;
        }
    if (i == -1)
        data[0] = val;
    ++size;
}

bool VertexSet::has_data(int val)
{
    for (int i = 0; i < size; ++i)
        if (data[i] == val)
            return true;
    return false;
}

int VertexSet::unordered_subtraction_size(const VertexSet& set0, const VertexSet& set1, int size_after_restrict)
{
    int size0 = set0.get_size();
    int size1 = set1.get_size();
    if (size_after_restrict != -1)
        size0 = size_after_restrict;

    int ret = size0;
    const int* set0_ptr = set0.get_data_ptr();
    for (int j = 0; j < size1; ++j)
        if (std::binary_search(set0_ptr, set0_ptr + size0, set1.get_data(j)))
            --ret;
    return ret;
}
