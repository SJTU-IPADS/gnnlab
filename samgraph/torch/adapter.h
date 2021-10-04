#ifndef SAMGRAPH_TORCH_OPS_H
#define SAMGRAPH_TORCH_OPS_H

#include <TH/TH.h>
#include <THC/THC.h>

#include <cstdint>

namespace samgraph {
namespace torch {

extern "C" {

THCudaTensor samgraph_torch_get_graph_feat(uint64_t key);
THCudaTensor samgraph_torch_get_graph_label(uint64_t key);
THCudaTensor samgraph_torch_get_graph_row(uint64_t key, int layer_idx);
THCudaTensor samgraph_torch_get_graph_col(uint64_t key, int layer_idx);
THCudaTensor samgraph_torch_get_graph_data(uint64_t key, int layer_idx);

THTensor samgraph_torch_get_dataset_feat();
THTensor samgraph_torch_get_dataset_label();
THCudaTensor samgraph_torch_get_graph_input_nodes(uint64_t key);
THCudaTensor samgraph_torch_get_graph_output_nodes(uint64_t key);

}

}  // namespace torch
}  // namespace samgraph

#endif  // SAMGRAPH_TORCH_OPS_H