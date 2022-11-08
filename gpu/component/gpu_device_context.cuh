#pragma once

#include <cstdint>

#include "graph.h"
#include "labeled_graph.h"

struct GPUDeviceContext {
    int32_t dev_id;
};

struct GraphDeviceContext : public GPUDeviceContext {
    v_index_t *dev_edge, *dev_edge_from;
    e_index_t *dev_vertex;
    uint32_t *dev_tmp;
    const Graph *g;
};

struct LabeledGraphDeviceContext : public GPUDeviceContext {
    uint32_t *dev_edge;
    uint32_t *dev_labeled_vertex;
    int32_t *dev_v_label;
    const LabeledGraph *g;
};