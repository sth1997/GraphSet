#include "dataloader.h"
#include "graph.h"
#include "vertex_set.h"
#include "common.h"
#include <cstdlib>
#include <cstring>

struct FileGuard {
    FILE *fp;

    FileGuard(FILE* p) : fp(p) {}
    ~FileGuard() { fclose(fp); }
};

struct GraphHeader {
    uint32_t v_cnt, e_cnt, max_intersection_size, checksum;
    int64_t tri_cnt;
};

static void calculate_checksum(GraphHeader& h)
{
    h.checksum = 0;
    uint32_t sum = 0;
    uint32_t *p = reinterpret_cast<uint32_t*>(&h);
    int len = sizeof(h) / sizeof(uint32_t);
    for (int i = 0; i < len; ++i)
        sum += p[i];
    h.checksum = -sum;
}

static bool do_checksum(GraphHeader& h)
{
    uint32_t sum = 0;
    uint32_t *p = reinterpret_cast<uint32_t*>(&h);
    int len = sizeof(h) / sizeof(uint32_t);
    for (int i = 0; i < len; ++i)
        sum += p[i];
    return sum == 0;
}

/**
 * @brief dump graph data into a file
 * @note for internal use only
 * @return true if graph successfully dumped
 */
static bool dump_graph(Graph& g, const char* filename)
{
    GraphHeader h;
    h.v_cnt = g.v_cnt;
    h.e_cnt = g.e_cnt;
    h.tri_cnt = g.tri_cnt;
    h.max_intersection_size = VertexSet::max_intersection_size; // ...
    calculate_checksum(h);

    FILE *fp = fopen(filename, "wb");
    if (!fp)
        return false;
    fwrite(&h, sizeof(h), 1, fp);
    fwrite(g.vertex, sizeof(uint32_t), g.v_cnt + 1, fp);
    fwrite(g.edge, sizeof(uint32_t), g.e_cnt, fp);
    fclose(fp);
    return true;
}

/**
 * @brief load graph from a file
 * @note for internal use only
 * @return true if graph is successfully loaded
 */
static bool load_graph(Graph& g, const char* filename)
{
    GraphHeader h;

    FILE *fp = fopen(filename, "rb");
    if (!fp)
        return false;

    FileGuard guard(fp);
    if (fread(&h, sizeof(h), 1, fp) != 1) {
        printf("load_graph: bad header.\n");
        return false;
    }
    if (!do_checksum(h)) {
        printf("load_graph: checksum != 0. stop.\n");
        return false;
    }

    g.v_cnt = h.v_cnt;
    g.e_cnt = h.e_cnt;
    g.tri_cnt = h.tri_cnt;
    VertexSet::max_intersection_size = h.max_intersection_size; // ...

    g.vertex = new unsigned int[h.v_cnt + 1];
    g.edge = new int[g.e_cnt];

    if (fread(g.vertex, sizeof(uint32_t), h.v_cnt + 1, fp) != h.v_cnt + 1) {
        printf("load_graph: failed to load vertexes.\n");
        return false;
    }
    if (fread(g.edge, sizeof(uint32_t), h.e_cnt, fp) != h.e_cnt) {
        printf("load_graph: failed to load edges.\n");
        return false;
    }
    return true;
}

bool DataLoader::fast_load(Graph* &g, const char* path)
{
    g = new Graph();
    bool success = load_graph(*g, path);
    if (!success)
        delete g;
    return success;
}

bool DataLoader::load_data(Graph* &g, DataType type, const char* path, bool binary_input, int oriented_type) {
    if(type == Patents || type == Orkut || type == complete8 || type == LiveJournal || type == MiCo || type == CiteSeer || type == Wiki_Vote) {
        return general_load_data(g, type, path, binary_input, oriented_type);
    }

    if( type == Twitter) {
        return twitter_load_data(g, type, path, oriented_type);
    }
    printf("invalid DataType!\n");
    return false;
}

static inline bool read_u32_pair(FILE* fp, bool binary, uint32_t& u, uint32_t& v)
{
    if (!binary) {
        return fscanf(fp, "%u%u", &u, &v) == 2;
    } else {
        bool f1 = (fread(&u, sizeof(uint32_t), 1, fp) == 1);
        bool f2 = (fread(&v, sizeof(uint32_t), 1, fp) == 1);
        return f1 && f2;
    }
}

bool DataLoader::general_load_data(Graph* &g, DataType type, const char* path, bool binary, int oriented_type) {
    FILE *fp = fopen(path, binary ? "rb" : "r");

    if (!fp) {
        printf("File %s not found.\n", path);
        return false;
    }

    FileGuard guard(fp);
    printf("Load begin in %s\n", path);
    g = new Graph();

    //load triangle counting information
    switch(type) {
        case DataType::Patents : {
            g->tri_cnt = Patents_tri_cnt;
            break;
        }
        case DataType::LiveJournal : {
            g->tri_cnt = LiveJournal_tri_cnt;
            break;
        }
        case DataType::MiCo : {
            g->tri_cnt = MiCo_tri_cnt;
            break;
        }
        case DataType::CiteSeer : {
            g->tri_cnt = CiteSeer_tri_cnt;
            break;
        }
        case DataType::Wiki_Vote : {
            g->tri_cnt = Wiki_Vote_tri_cnt;
            break;
        }
        case DataType::Orkut : {
            g->tri_cnt = Orkut_tri_cnt;
            break;
        }
        default : {
            g->tri_cnt = -1;
            break;
        }
    }

    uint32_t x, y, tmp_v, tmp_e;
    read_u32_pair(fp, binary, x, y);
    g->v_cnt = x, g->e_cnt = y * 2; 

    int *degree = new int[g->v_cnt];
    memset(degree, 0, g->v_cnt * sizeof(int));
    auto e = new std::pair<uint32_t, uint32_t>[g->e_cnt];
    id.clear();

    tmp_v = 0;
    tmp_e = 0;
    while (read_u32_pair(fp, binary, x, y)) {
        if(x == y) {
            printf("find self circle\n");
            g->e_cnt -=2;
            continue;
            //return false;
        }
        if(!id.count(x)) id[x] = tmp_v ++;
        if(!id.count(y)) id[y] = tmp_v ++;
        x = id[x];
        y = id[y];
        e[tmp_e++] = std::make_pair(x,y);
        e[tmp_e++] = std::make_pair(y,x);
        ++degree[x];
        ++degree[y];
        if (tmp_e % 1000000 == 0) {
            printf("%u edges loaded\n",tmp_e);
            fflush(stdout);
        }
    }

    // oriented_type == 0 do nothing
    //               == 1 high degree first
    //               == 2 low degree first
    if ( oriented_type != 0 ) {
        std::pair<int,int> *rank = new std::pair<int,int>[g->v_cnt];
        int *new_id = new int[g->v_cnt];
        for(int i = 0; i < g->v_cnt; ++i) rank[i] = std::make_pair(i,degree[i]);
        if( oriented_type == 1) std::sort(rank, rank + g->v_cnt, cmp_degree_gt);
        if( oriented_type == 2) std::sort(rank, rank + g->v_cnt, cmp_degree_lt);
        for(int i = 0; i < g->v_cnt; ++i) new_id[rank[i].first] = i;
        for(unsigned int i = 0; i < g->e_cnt; ++i) {
            e[i].first = new_id[e[i].first];
            e[i].second = new_id[e[i].second];
        }
        delete[] rank;
        delete[] new_id;
    }
    //TODO we only want to get the second max degree, don't need to sort the whole array.
    // std::sort(degree, degree + g->v_cnt);
    std::nth_element(degree, degree + g->v_cnt - 2, degree + g->v_cnt); // OK, fixed

    // The max size of intersections is the second largest degree.
    //TODO VertexSet::max_intersection_size has different value with different dataset, but we use a static variable now.
    VertexSet::max_intersection_size = std::max( VertexSet::max_intersection_size, degree[g->v_cnt - 2]);
    delete[] degree;
    if(tmp_v != g->v_cnt) {
        printf("vertex number error!\n");
    }
    if(tmp_e != g->e_cnt) {
        printf("edge number error!\n");
    }
    if(tmp_v != g->v_cnt || tmp_e != g->e_cnt) {
        delete g;
        delete[] e;
        return false;
    }
    std::sort(e,e+tmp_e,cmp_pair);
    g->e_cnt = std::unique(e,e+tmp_e) - e;
    for(unsigned int i = 0; i < g->e_cnt - 1; ++i)
        if(e[i] == e[i+1]) {
            printf("have same edge\n");
            delete g;
            delete[] e;
            return false;
        }
    g->edge = new int[g->e_cnt];
    g->vertex = new unsigned int[g->v_cnt + 1];
    bool* have_edge = new bool[g->v_cnt];
    int lst_v = -1;
    memset(have_edge, 0, g->v_cnt * sizeof(bool));
    for(unsigned int i = 0; i < g->e_cnt; ++i) {
        if(e[i].first != lst_v) {
            have_edge[e[i].first] = true;
            g->vertex[e[i].first] = i;
        }
        lst_v = e[i].first;
        g->edge[i] = e[i].second;
    }
    delete[] e;
    printf("Success! There are %d nodes and %u edges.\n",g->v_cnt,g->e_cnt);
    fflush(stdout);
    g->vertex[g->v_cnt] = g->e_cnt;
    for(int i = g->v_cnt - 1; i >= 0; --i)
        if(!have_edge[i]) {
            g->vertex[i] = g->vertex[i+1];
        }
    delete[] have_edge;

    bool ok = dump_graph(*g, "patents.g");
    printf("dump graph %d\n", ok);

    return true;
}

bool DataLoader::twitter_load_data(Graph *&g, DataType type, const char* path, int oriented_type) {
    if (freopen(path, "r", stdin) == NULL)
    {
        printf("File not found. %s\n", path);
        return false;
    }
    printf("Load begin in %s\n",path);
    g = new Graph();
    g->tri_cnt = Twitter_tri_cnt;
    unsigned int* buffer = new unsigned int[41652230u + 2936729768u + 10u];
    FILE* file = fopen(path, "r");
    fread(buffer, sizeof(unsigned int), 41652230u + 2936729768u + 4, file);
    g->v_cnt = buffer[0];
    g->e_cnt = buffer[1];
    int mx_degree = buffer[2];
    VertexSet::max_intersection_size = std::max( VertexSet::max_intersection_size, mx_degree);
    g->edge = new int [g->e_cnt];
    g->vertex = new unsigned int [g->v_cnt + 1];
    for(int i = 0; i < g->v_cnt + 1; ++i)
        g->vertex[i] = buffer[ 3 + i];
    for(unsigned int i = 0; i < g->e_cnt; ++i)
        g->edge[i] = buffer[4 + g->v_cnt + i];
    delete[] buffer;
    return true;
}

/**
 * @note：好奇怪啊。已经是完全图了，有些操作是多余的。
 */
bool DataLoader::load_data(Graph* &g, int clique_size) {
    g = new Graph();

    g->v_cnt = clique_size;
    g->e_cnt = clique_size * (clique_size - 1) / 2;

    int* degree = new int[g->v_cnt];
    memset(degree, 0, g->v_cnt * sizeof(int));
    g->e_cnt *= 2;
    std::pair<int,int> *e = new std::pair<int,int>[g->e_cnt];
    id.clear();
    int tmp_v;
    unsigned int tmp_e;
    tmp_v = 0;
    tmp_e = 0;
    for(int i = 0; i < clique_size; ++i)
        for(int j = 0; j < i; ++j) {
            int x = i, y = j;
            if(!id.count(x)) id[x] = tmp_v ++;
            if(!id.count(y)) id[y] = tmp_v ++;
            x = id[x];
            y = id[y];
            e[tmp_e++] = std::make_pair(x,y);
            e[tmp_e++] = std::make_pair(y,x);
            ++degree[x];
            ++degree[y];
        }

    std::sort(degree, degree + g->v_cnt);

    // The max size of intersections is the second largest degree.
    //TODO VertexSet::max_intersection_size has different value with different dataset, but we use a static variable now.
    VertexSet::max_intersection_size = std::max( VertexSet::max_intersection_size, degree[g->v_cnt - 2]);
    delete[] degree;
    if(tmp_v != g->v_cnt) {
        printf("vertex number error!\n");
    }
    if(tmp_e != g->e_cnt) {
        printf("edge number error!\n");
    }
    if(tmp_v != g->v_cnt || tmp_e != g->e_cnt) {
        fclose(stdin);
        delete g;
        delete[] e;
        return false;
    }
    std::sort(e,e+tmp_e,cmp_pair);
    g->e_cnt = unique(e,e+tmp_e) - e;
    g->edge = new int[g->e_cnt];
    g->vertex = new unsigned int[g->v_cnt + 1];
    bool* have_edge = new bool[g->v_cnt];
    int lst_v = -1;
    memset(have_edge, 0, g->v_cnt * sizeof(bool));
    for(unsigned int i = 0; i < g->e_cnt; ++i) {
        if(e[i].first != lst_v) {
            have_edge[e[i].first] = true;
            g->vertex[e[i].first] = i;
        }
        lst_v = e[i].first;
        g->edge[i] = e[i].second;
    }
    delete[] e;
    g->vertex[g->v_cnt] = g->e_cnt;
    for(int i = g->v_cnt - 1; i >= 0; --i)
        if(!have_edge[i]) {
            g->vertex[i] = g->vertex[i+1];
        }
    delete[] have_edge;
    return true;
}

bool DataLoader::load_complete(Graph* &g, int clique_size) {
    g = new Graph();

    g->v_cnt = clique_size;
    g->e_cnt = clique_size * (clique_size - 1) / 2;

    int* degree = new int[g->v_cnt];
    memset(degree, 0, g->v_cnt * sizeof(int));
    g->e_cnt *= 2;
    std::pair<int,int> *e = new std::pair<int,int>[g->e_cnt];
    id.clear();
    int tmp_v;
    unsigned int tmp_e;
    tmp_v = 0;
    tmp_e = 0;
    for(int i = 0; i < clique_size; ++i)
        for(int j = 0; j < i; ++j) {
            int x = i, y = j;
            if(!id.count(x)) id[x] = tmp_v ++;
            if(!id.count(y)) id[y] = tmp_v ++;
            x = id[x];
            y = id[y];
            e[tmp_e++] = std::make_pair(x,y);
            e[tmp_e++] = std::make_pair(y,x);
            ++degree[x];
            ++degree[y];
        }

    std::sort(degree, degree + g->v_cnt);

    // The max size of intersections is the second largest degree.
    //TODO VertexSet::max_intersection_size has different value with different dataset, but we use a static variable now.
    VertexSet::max_intersection_size = std::max( VertexSet::max_intersection_size, degree[g->v_cnt - 2]);
    //g->max_degree = degree[g->v_cnt - 1];
    delete[] degree;
    if(tmp_v != g->v_cnt) {
        printf("vertex number error!\n");
    }
    if(tmp_e != g->e_cnt) {
        printf("edge number error!\n");
    }
    if(tmp_v != g->v_cnt || tmp_e != g->e_cnt) {
        fclose(stdin);
        delete g;
        delete[] e;
        return false;
    }
    std::sort(e,e+tmp_e,cmp_pair);
    g->e_cnt = unique(e,e+tmp_e) - e;
    g->edge = new int[g->e_cnt];
    g->vertex = new unsigned int[g->v_cnt + 1];
    bool* have_edge = new bool[g->v_cnt];
    int lst_v = -1;
    for(int i = 0; i < g->v_cnt; ++i) have_edge[i] = false;
    for(unsigned int i = 0; i < g->e_cnt; ++i) {
        if(e[i].first != lst_v) {
            have_edge[e[i].first] = true;
            g->vertex[e[i].first] = i;
        }
        lst_v = e[i].first;
        g->edge[i] = e[i].second;
    }
    delete[] e;
    g->vertex[g->v_cnt] = g->e_cnt;
    for(int i = g->v_cnt - 1; i >= 0; --i)
        if(!have_edge[i]) {
            g->vertex[i] = g->vertex[i+1];
        }
    delete[] have_edge;
    return true;
}

bool DataLoader::cmp_pair(std::pair<int,int>a, std::pair<int,int>b) {
    return a.first < b.first || (a.first == b.first && a.second < b.second);
}

bool DataLoader::cmp_degree_gt(std::pair<int,int> a,std::pair<int,int> b) {
    return a.second > b.second;
}

bool DataLoader::cmp_degree_lt(std::pair<int,int> a,std::pair<int,int> b) {
    return a.second < b.second;
}

long long DataLoader::comb(int n, int k) {
    long long ans = 1;
    for(int i = n; i > n - k; --i)
        ans = ans * i;
    for(int i = 1; i <= k; ++i)
        ans = ans / k;
    return ans;
}
