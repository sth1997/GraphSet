#include "dataloader.h"
#include "graph.h"
#include "labeled_graph.h"
#include "vertex_set.h"
#include "common.h"
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

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
    printf("load_graph: %u vertexes, %u edges\n", g.v_cnt, g.e_cnt);
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

bool DataLoader::load_labeled_data(LabeledGraph* &g, DataType type, const char* path) {
    if(type == Patents || type == Orkut || type == complete8 || type == LiveJournal || type == MiCo || type == CiteSeer || type == Wiki_Vote || type == Twitter) {
        return general_load_labeled_data(g, type, path);
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

//默认节点编号从0~cnt-1，不进行重排序；默认同一条边在输入数据中会出现正反各一次（所以按照单向边读入）
//第一行为点数v_cnt
//接下来v_cnt行，每行第一个数是点编号，第二个数是label（可以不从0开始，会重新映射），之后一直到行末是邻居节点编号
bool DataLoader::general_load_labeled_data(LabeledGraph* &g, DataType type, const char* path) {
    std::ifstream fin(path);
    if (!fin) {
        printf("File %s not found.\n", path);
        return false;
    }

    printf("Load begin in %s\n", path);
    g = new LabeledGraph();

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

    uint32_t x, y, tmp_v, tmp_e, tmp_l;
    fin >> x;
    g->v_cnt = x;
    //g->l_cnt = y * 2; 

    int *degree = new int[g->v_cnt];
    g->v_label = new int[g->v_cnt];
    g->label_frequency = new int[g->v_cnt];
    memset(g->label_frequency, 0, sizeof(int) * g->v_cnt);
    memset(degree, 0, g->v_cnt * sizeof(int));
    auto e = new std::vector< std::pair<uint32_t, uint32_t> >;
    label.clear();
    //id.clear();

    tmp_v = 0;
    tmp_e = 0;
    tmp_l = 0;

    std::string line_str;
    std::stringstream ss;
    while (fin >> x >> y) {
        ++tmp_v;
        if (!label.count(y)) label[y] = tmp_l ++;
        g->v_label[x] = label[y];
        g->label_frequency[label[y]] += 1;
        std::getline(fin, line_str);
        ss.clear();
        ss << line_str;
        int v;
        while (ss >> v)
        {
            /*char ch;
            for (fscanf(fp, "%c", &ch); (ch < '0' || ch > '9') && ch != '\n'; fscanf(fp, "%c", &ch));
            if (ch == '\n')
                break;
            int v = ch - '0';
            for (fscanf(fp, "%c", &ch); ch >= '0' && ch <= '9'; fscanf(fp, "%c", &ch))
                v = v * 10 + ch - '0';
            */
            
            if(x == v) {
                printf("find self circle\n");
                //g->e_cnt -=2;
                continue;
            }
            //if(!id.count(x)) id[x] = tmp_v ++;
            //if(!id.count(y)) id[y] = tmp_v ++;
            //x = id[x];
            //y = id[y];
            //e[tmp_e++] = std::make_pair(x,y);
            //e[tmp_e++] = std::make_pair(y,x);
            e->push_back(std::make_pair(x,v));//按照单向边处理
            ++degree[x];
            //++degree[y];
            if (e->size() % 1000000 == 0) {
                printf("%u edges loaded\n",e->size());
                fflush(stdout);
            }
        }
    }
    g->l_cnt = tmp_l;

    std::nth_element(degree, degree + g->v_cnt - 2, degree + g->v_cnt);

    // The max size of intersections is the second largest degree.
    //TODO VertexSet::max_intersection_size has different value with different dataset, but we use a static variable now.
    VertexSet::max_intersection_size = std::max( VertexSet::max_intersection_size, degree[g->v_cnt - 2]);
    delete[] degree;
    if(tmp_v != g->v_cnt) {
        printf("vertex number error!\n");
        delete g;
        delete[] e;
        return false;
    }

    //将点按照label重排序（这样子在pattern matching第一层for循环的时候就不需要判断label了）
    std::pair<int,int> *rank = new std::pair<int,int>[g->v_cnt];
    int *new_id = new int[g->v_cnt];
    for(int i = 0; i < g->v_cnt; ++i) rank[i] = std::make_pair(i,g->v_label[i]);
    std::sort(rank, rank + g->v_cnt, cmp_label);
    for(int i = 0; i < g->v_cnt; ++i) {
        new_id[rank[i].first] = i;
        g->v_label[i] = rank[i].second;
    }
    for(auto& edge : *e) {
        edge.first = new_id[edge.first];
        edge.second = new_id[edge.second];
    }
    delete[] rank;
    delete[] new_id;
    g->label_start_idx = new unsigned int[g->l_cnt + 1];
    memset(g->label_start_idx, -1, sizeof(unsigned int) * (g->l_cnt + 1));
    g->label_start_idx[0] = 0;
    g->label_start_idx[g->l_cnt] = g->v_cnt;
    for (int i = 1; i < g->v_cnt; ++i)
        if (g->v_label[i] != g->v_label[i - 1])
            g->label_start_idx[g->v_label[i]] = i;
    for (int i = g->l_cnt - 1; i > 0; --i)
        if (g->label_start_idx[i] == -1)
            g->label_start_idx[i] = g->label_start_idx[i + 1];

    /*if(tmp_e != g->e_cnt) {
        printf("edge number error!\n");
    }*/
    /*if(tmp_v != g->v_cnt || tmp_e != g->e_cnt) {
        delete g;
        delete[] e;
        return false;
    }*/
    auto labeled_e = new std::vector< std::tuple<uint32_t, uint32_t, uint32_t> >;
    for (auto& p: *e)
        labeled_e->push_back(std::make_tuple(p.first, g->v_label[p.second], p.second));
    std::sort(labeled_e->begin(), labeled_e->end(), cmp_tuple);
    printf("e_cnt = %d\n", e->size());
    fflush(stdout);
    g->e_cnt = e->size();
    /*g->e_cnt = std::unique(labeled_e,e+tmp_e) - e;
    for(unsigned int i = 0; i < g->e_cnt - 1; ++i)
        if(e[i] == e[i+1]) {
            printf("have same edge\n");
            delete g;
            delete[] e;
            return false;
        }*/
    g->edge = new int[g->e_cnt];
    g->labeled_vertex = new unsigned int[g->v_cnt * g->l_cnt + 1];
    bool* have_edge = new bool[g->v_cnt * g->l_cnt];
    int lst_v = -1;
    int lst_l = -1;
    memset(have_edge, 0, g->v_cnt * g->l_cnt * sizeof(bool));
    for(unsigned int i = 0; i < g->e_cnt; ++i) {
        if(std::get<0>(labeled_e->at(i)) != lst_v || std::get<1>(labeled_e->at(i)) != lst_l) {
            have_edge[std::get<0>(labeled_e->at(i)) * g->l_cnt + std::get<1>(labeled_e->at(i))] = true;
            g->labeled_vertex[std::get<0>(labeled_e->at(i)) * g->l_cnt + std::get<1>(labeled_e->at(i))] = i;
        }
        lst_v = std::get<0>(labeled_e->at(i));
        lst_l = std::get<1>(labeled_e->at(i));
        g->edge[i] = std::get<2>(labeled_e->at(i));
    }
    delete e;
    delete labeled_e;
    printf("Success! There are %d nodes and %u edges.\n",g->v_cnt,g->e_cnt);
    fflush(stdout);
    g->labeled_vertex[g->v_cnt * g->l_cnt] = g->e_cnt;
    for(int i = g->v_cnt * g->l_cnt - 1; i >= 0; --i)
        if(!have_edge[i]) {
            g->labeled_vertex[i] = g->labeled_vertex[i+1];
        }
    delete[] have_edge;

    g->label_map = label;

    //bool ok = dump_graph(*g, "patents.g");
    //printf("dump graph %d\n", ok);

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

bool DataLoader::cmp_tuple(std::tuple<int,int,int>a, std::tuple<int,int,int>b) {
    return std::get<0>(a) < std::get<0>(b) || (std::get<0>(a) == std::get<0>(b) && std::get<1>(a) < std::get<1>(b)) || (std::get<0>(a) == std::get<0>(b) && std::get<1>(a) == std::get<1>(b) && std::get<2>(a) < std::get<2>(b));
}

bool DataLoader::cmp_label(std::pair<int,int> a,std::pair<int,int> b) {
    return a.second < b.second;
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
