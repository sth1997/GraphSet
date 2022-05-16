#include <graph.h>
#include <dataloader.h>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <omp.h>

namespace my {

bool contains(const int d[], int n, int vt) {
    int mid, l = 0, r = n - 1;
    while (l <= r) {
        mid = (l + r) >> 1;
        if (d[mid] < vt) {
            l = mid + 1;
        } else if (d[mid] > vt) {
            r = mid - 1;
        } else {
            return true;
        }
    }
    return false;
}

bool contains(const std::vector<int>& v, int vt) {
    return contains(&v[0], v.size(), vt);
}

struct NeighborSet {
    struct Iterator {
        int *p;

        Iterator() {}
        Iterator(int *_p) : p{_p} {}

        Iterator& operator++() {
            ++p;
            return *this;
        }
        bool operator==(const Iterator& rhs) const { return p == rhs.p; }
        bool operator!=(const Iterator& rhs) const { return !(*this == rhs); }
        int& operator*() { return *p; }
        int* operator->() { return p; }
    };

    const Graph &g;
    int v;

    NeighborSet(const Graph& graph, int vertex) : g{graph}, v{vertex} {}
    int size() const { return g.vertex[v + 1] - g.vertex[v]; }
    Iterator begin() { return Iterator{&g.edge[g.vertex[v]]}; }
    Iterator end() { return Iterator{&g.edge[g.vertex[v + 1]]}; }

    bool contains(int vt) const {
        return my::contains(&g.edge[g.vertex[v]], size(), vt);
    }
};

NeighborSet N(const Graph& g, int v) { return NeighborSet{g, v}; }
int V(const Graph& g) { return g.v_cnt; }
}

using my::V;
using my::N;
using my::contains;

uint64_t count_wedge(const Graph& g) {
    uint64_t res = 0;
    // 0 -- 1 -- 2
    // for (int v0 = 0; v0 < V(g); ++v0) {
    //     auto s1 = N(g, v0);
    //     for (int v1 : s1) {
    //         auto s2 = N(g, v1);
    //         for (int v2 : s2) {
    //             if (v0 >= v2)
    //                 continue;
    //             if (!s1.contains(v2))
    //                 ++res;
    //         }
    //     }
    // }

    for (int v1 = 0; v1 < V(g); ++v1) {
        auto s02 = N(g, v1);
        for (int v0 : s02) {
            for (int v2 : s02) {
                if (v2 >= v0)
                    break;
                if (!N(g, v0).contains(v2))
                    ++res;
            }
        }
    }

    return res;
}

uint64_t count_square(const Graph& g) {
    uint64_t res = 0;
    /*
        0 -- 1
        |    |
        2 -- 3
    */
    #pragma omp parallel for reduction(+:res)
    for (int v0 = 0; v0 < V(g); ++v0) {
        auto s12 = N(g, v0);
        for (int v1 : s12) {
            if (v1 >= v0)
                break;

            for (int v2 : s12) {
                if (v2 >= v1)
                    break;
                
                auto s3_0 = N(g, v1);
                if (s3_0.contains(v2))
                    continue;
                
                auto s3_1 = N(g, v2);
                std::vector<int> s3;
                std::set_intersection(s3_0.begin(), s3_0.end(), s3_1.begin(), s3_1.end(),
                    std::back_inserter(s3));
                for (int v3 : s3) {
                    if (v3 >= v0)
                        break;
                    
                    if (!s12.contains(v3))
                        ++res;
                }
            }
        }
    }
    return res;
}

uint64_t count_diamond(const Graph& g) {
    uint64_t res = 0;
    /*
        0 - 2
        |\  |
        | \ |
        |  \|
        3 - 1
    */
    #pragma omp parallel for reduction(+:res)
    for (int v0 = 0; v0 < V(g); ++v0) {
        auto s1 = N(g, v0);
        for (int v1 : s1) {
            if (v1 >= v0)
                break;
            
            auto s23_0 = N(g, v1);
            std::vector<int> s23;
            std::set_intersection(s1.begin(), s1.end(), s23_0.begin(), s23_0.end(),
                std::back_inserter(s23));            
            for (int v2 : s23) {
                for (int v3 : s23) {
                    if (v3 >= v2)
                        break;
                    
                    if (!N(g, v2).contains(v3))
                        ++res;
                }
            }
        }
    }
    return res;
}

uint64_t count_tailed_triangle(const Graph& g) {
    uint64_t res = 0;
    /*
        1
        |\
        | 0 -- 3
        |/
        2
    */
    #pragma omp parallel for reduction(+:res)
    for (int v0 = 0; v0 < V(g); ++v0) {
        auto s13 = N(g, v0);
        for (int v1 : s13) {            
            auto s2_0 = N(g, v1);
            std::vector<int> s2;
            std::set_intersection(s13.begin(), s13.end(), s2_0.begin(), s2_0.end(),
                std::back_inserter(s2));            
            for (int v2 : s2) {
                if (v2 >= v1)
                    break;

                for (int v3 : s13) {                    
                    if (!N(g, v1).contains(v3) && !N(g, v2).contains(v3))
                        ++res;
                }
            }
        }
    }
    return res;
}

int main(int argc, char* argv[])
{
    Graph *pg;
    DataLoader d;

    d.fast_load(pg, argv[1]);
    // printf("wedge: %ld\n", count_wedge(*pg));
    // printf("square: %ld\n", count_square(*pg));
    // printf("diamond: %ld\n", count_diamond(*pg));
    printf("tailed-triangle: %ld\n", count_tailed_triangle(*pg));

    return 0;
}
