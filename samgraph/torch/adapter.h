/*
 * Copyright 2022 Institute of Parallel and Distributed Systems, Shanghai Jiao Tong University
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#ifndef SAMGRAPH_TORCH_OPS_H
#define SAMGRAPH_TORCH_OPS_H

// #include <TH/TH.h>
// #include <THC/THC.h>
#include <torch/torch.h>

#include <cstdint>

namespace samgraph {
namespace torch {

extern "C" {

::torch::Tensor samgraph_torch_get_graph_feat(uint64_t key);
::torch::Tensor samgraph_torch_get_graph_label(uint64_t key);
::torch::Tensor samgraph_torch_get_graph_row(uint64_t key, int layer_idx);
::torch::Tensor samgraph_torch_get_graph_col(uint64_t key, int layer_idx);
::torch::Tensor samgraph_torch_get_graph_data(uint64_t key, int layer_idx);
::torch::Tensor samgraph_torch_get_unsupervised_graph_row(uint64_t key);
::torch::Tensor samgraph_torch_get_unsupervised_graph_col(uint64_t key);

::torch::Tensor samgraph_torch_get_dataset_feat();
::torch::Tensor samgraph_torch_get_dataset_label();
::torch::Tensor samgraph_torch_get_graph_input_nodes(uint64_t key);
::torch::Tensor samgraph_torch_get_graph_output_nodes(uint64_t key);

}

}  // namespace torch
}  // namespace samgraph

#endif  // SAMGRAPH_TORCH_OPS_H