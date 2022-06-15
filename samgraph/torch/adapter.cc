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

#include "adapter.h"

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include "torch/types.h"

#undef LOG
#undef CHECK_NOTNULL
#undef CHECK_NE
#undef CHECK_LE
#undef CHECK_GE
#undef CHECK_LT
#undef CHECK_GT
#undef CHECK_EQ
#undef CHECK
#include "../common/common.h"
#include "../common/cuda/cuda_engine.h"
#include "../common/dist/dist_engine.h"
#include "../common/profiler.h"
#include "../common/timer.h"

// Use the torch built-in CHECK macros
// #include "../common/logging.h"

namespace {
::torch::Dtype to_torch_data_type(samgraph::common::DataType dt) {
  switch (dt) {
    case samgraph::common::kF32: return ::torch::kF32;
    case samgraph::common::kF64: return ::torch::kF64;
    case samgraph::common::kF16: return ::torch::kF16;
    case samgraph::common::kU8:  return ::torch::kU8;
    case samgraph::common::kI32: return ::torch::kI32;
    case samgraph::common::kI8:  return ::torch::kI8;
    case samgraph::common::kI64: return ::torch::kI64;
  }
}
};

namespace samgraph {
namespace torch {

::torch::Tensor GetGraphFeature(uint64_t key) {
  auto graph_batch = common::Engine::Get()->GetGraphBatch();
  auto feat = graph_batch->input_feat;
  auto trainer_ctx = common::Engine::Get()->GetTrainerCtx();
  auto device = "cuda:" + std::to_string(trainer_ctx.device_id);

  CHECK_EQ(key, graph_batch->key);

  ::torch::Tensor tensor = ::torch::from_blob(
      feat->MutableData(),
      {(long long)feat->Shape()[0], (long long)feat->Shape()[1]},
      [feat](void* data) {},
      ::torch::TensorOptions().dtype(to_torch_data_type(feat->Type())).device(device));

  return tensor;
}

::torch::Tensor GetGraphLabel(uint64_t key) {
  auto graph_batch = common::Engine::Get()->GetGraphBatch();
  auto label = graph_batch->output_label;
  auto trainer_ctx = common::Engine::Get()->GetTrainerCtx();
  auto device = "cuda:" + std::to_string(trainer_ctx.device_id);

  CHECK_EQ(key, graph_batch->key);
  ::torch::Tensor tensor = ::torch::from_blob(
      label->MutableData(), {(long long)label->Shape()[0]},
      [label](void* data) {},
      ::torch::TensorOptions().dtype(to_torch_data_type(label->Type())).device(device));

  return tensor;
}

::torch::Tensor GetGraphRow(uint64_t key, int layer_idx) {
  auto graph_batch = common::Engine::Get()->GetGraphBatch();
  auto row = graph_batch->graphs[layer_idx]->row;
  auto trainer_ctx = common::Engine::Get()->GetTrainerCtx();
  auto device = "cuda:" + std::to_string(trainer_ctx.device_id);

  CHECK_EQ(key, graph_batch->key);
  ::torch::Tensor tensor = ::torch::from_blob(
      row->MutableData(), {(long long)row->Shape()[0]}, [row](void* data) {},
      ::torch::TensorOptions().dtype(::torch::kI32).device(device));

  return tensor;
}

::torch::Tensor GetGraphCol(uint64_t key, int layer_idx) {
  auto graph_batch = common::Engine::Get()->GetGraphBatch();
  auto col = graph_batch->graphs[layer_idx]->col;
  auto trainer_ctx = common::Engine::Get()->GetTrainerCtx();
  auto device = "cuda:" + std::to_string(trainer_ctx.device_id);

  CHECK_EQ(key, graph_batch->key);
  ::torch::Tensor tensor = ::torch::from_blob(
      col->MutableData(), {(long long)col->Shape()[0]}, [col](void* data) {},
      ::torch::TensorOptions().dtype(::torch::kI32).device(device));

  return tensor;
}

::torch::Tensor GetUnsupervisedGraphRow(uint64_t key) {
  auto graph_batch = common::Engine::Get()->GetGraphBatch();
  CHECK_NE(graph_batch, nullptr);
  CHECK_NE(graph_batch->unsupervised_graph, nullptr);
  CHECK_NE(graph_batch->unsupervised_graph->row, nullptr);
  auto row = graph_batch->unsupervised_graph->row;
  auto trainer_ctx = common::Engine::Get()->GetTrainerCtx();
  auto device = "cuda:" + std::to_string(trainer_ctx.device_id);

  CHECK_EQ(key, graph_batch->key);
  ::torch::Tensor tensor = ::torch::from_blob(
      row->MutableData(), {(long long)row->Shape()[0]}, [row](void* data) {},
      ::torch::TensorOptions().dtype(::torch::kI32).device(device));

  return tensor;
}

::torch::Tensor GetUnsupervisedGraphCol(uint64_t key) {
  auto graph_batch = common::Engine::Get()->GetGraphBatch();
  CHECK_NE(graph_batch, nullptr);
  CHECK_NE(graph_batch->unsupervised_graph, nullptr);
  CHECK_NE(graph_batch->unsupervised_graph->col, nullptr);
  auto col = graph_batch->unsupervised_graph->col;
  auto trainer_ctx = common::Engine::Get()->GetTrainerCtx();
  auto device = "cuda:" + std::to_string(trainer_ctx.device_id);

  CHECK_EQ(key, graph_batch->key);
  ::torch::Tensor tensor = ::torch::from_blob(
      col->MutableData(), {(long long)col->Shape()[0]}, [col](void* data) {},
      ::torch::TensorOptions().dtype(::torch::kI32).device(device));

  return tensor;
}

::torch::Tensor GetGraphData(uint64_t key, int layer_idx) {
  auto graph_batch = common::Engine::Get()->GetGraphBatch();
  auto data = graph_batch->graphs[layer_idx]->data;
  auto trainer_ctx = common::Engine::Get()->GetTrainerCtx();
  auto device = "cuda:" + std::to_string(trainer_ctx.device_id);

  CHECK_EQ(key, graph_batch->key);
  ::torch::Tensor tensor = ::torch::from_blob(
      data->MutableData(), {(long long)data->Shape()[0]}, [data](void* ) {},
      ::torch::TensorOptions().dtype(::torch::kI32).device(device));

  return tensor;
}

::torch::Tensor GetDatasetFeature() {
  auto feat = common::Engine::Get()->GetGraphDataset()->feat;

  CHECK(feat->Ctx().device_type == common::kCPU ||
        feat->Ctx().device_type == common::kMMAP);

  ::torch::Tensor tensor = ::torch::from_blob(
      feat->MutableData(),
      {(long long)feat->Shape()[0], (long long)feat->Shape()[1]},
      [feat](void* data) {},
      ::torch::TensorOptions().dtype(to_torch_data_type(feat->Type())).device("cpu"));

  return tensor;
}

::torch::Tensor GetDatasetLabel() {
  auto label = common::Engine::Get()->GetGraphDataset()->label;

  CHECK(label->Ctx().device_type == common::kCPU ||
        label->Ctx().device_type == common::kMMAP);

  ::torch::Tensor tensor = ::torch::from_blob(
      label->MutableData(), {(long long)label->Shape()[0]},
      [label](void* data) {},
      ::torch::TensorOptions().dtype(::torch::kI64).device("cpu"));

  return tensor;
}

::torch::Tensor GetGraphInputNodes(uint64_t key) {
  auto graph_batch = common::Engine::Get()->GetGraphBatch();
  auto input_nodes = graph_batch->input_nodes;
  auto sampler_ctx = common::Engine::Get()->GetSamplerCtx();
  auto device = "cuda:" + std::to_string(sampler_ctx.device_id);

  CHECK_EQ(key, graph_batch->key);
  ::torch::Tensor tensor = ::torch::from_blob(
      input_nodes->MutableData(), {(long long)input_nodes->Shape()[0]},
      [input_nodes](void* data) {},
      ::torch::TensorOptions().dtype(::torch::kI32).device(device));

  return tensor;
}

::torch::Tensor GetGraphOuputNodes(uint64_t key) {
  auto graph_batch = common::Engine::Get()->GetGraphBatch();
  auto output_nodes = graph_batch->output_nodes;
  auto sampler_ctx = common::Engine::Get()->GetSamplerCtx();
  auto device = "cuda:" + std::to_string(sampler_ctx.device_id);

  CHECK_EQ(key, graph_batch->key);
  ::torch::Tensor tensor = ::torch::from_blob(
      output_nodes->MutableData(), {(long long)output_nodes->Shape()[0]},
      [output_nodes](void* data) {},
      ::torch::TensorOptions().dtype(::torch::kI32).device(device));

  return tensor;
}

PYBIND11_MODULE(c_lib, m) {
  m.def("samgraph_torch_get_graph_feat", &GetGraphFeature);
  m.def("samgraph_torch_get_graph_label", &GetGraphLabel);
  m.def("samgraph_torch_get_graph_row", &GetGraphRow);
  m.def("samgraph_torch_get_graph_col", &GetGraphCol);
  m.def("samgraph_torch_get_graph_data", &GetGraphData);

  m.def("samgraph_torch_get_unsupervised_graph_row", &GetUnsupervisedGraphRow);
  m.def("samgraph_torch_get_unsupervised_graph_col", &GetUnsupervisedGraphCol);

  m.def("samgraph_torch_get_dataset_feat", &GetDatasetFeature);
  m.def("samgraph_torch_get_dataset_label", &GetDatasetLabel);
  m.def("samgraph_torch_get_graph_input_nodes", &GetGraphInputNodes);
  m.def("samgraph_torch_get_graph_output_nodes", &GetGraphOuputNodes);
}

}  // namespace torch
}  // namespace samgraph
