#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include "../common/common.h"
#include "../common/cuda/cuda_engine.h"
#include "../common/logging.h"
#include "../common/profiler.h"
#include "../common/timer.h"
#include "ops.h"

namespace samgraph {
namespace torch {

::torch::Tensor GetGraphFeature(uint64_t key) {
  auto graph_batch = common::Engine::Get()->GetGraphBatch();
  auto feat = graph_batch->input_feat;
  auto device = common::Engine::Get()->GetTrainDevice();
  auto device_str = "cuda:" + std::to_string(device);

  CHECK_EQ(key, graph_batch->key);
  LOG(DEBUG) << "GetGraphFeature feat shape " << feat->shape().size()
             << " dim0: " << feat->shape()[0] << " dim1: " << feat->shape()[1]
             << "device_str " << device_str;

  ::torch::Tensor tensor = ::torch::from_blob(
      feat->mutable_data(),
      {(long long)feat->shape()[0], (long long)feat->shape()[1]},
      [feat](void* data) {},
      ::torch::TensorOptions().dtype(::torch::kF32).device(device_str));

  return tensor;
}

::torch::Tensor GetGraphLabel(uint64_t key) {
  auto graph_batch = common::Engine::Get()->GetGraphBatch();
  auto label = graph_batch->output_label;
  auto device = common::Engine::Get()->GetTrainDevice();
  auto device_str = "cuda:" + std::to_string(device);

  CHECK_EQ(key, graph_batch->key);
  ::torch::Tensor tensor = ::torch::from_blob(
      label->mutable_data(), {(long long)label->shape()[0]},
      [label](void* data) {},
      ::torch::TensorOptions().dtype(::torch::kI64).device(device_str));

  return tensor;
}

::torch::Tensor GetGraphRow(uint64_t key, int layer_idx) {
  auto graph_batch = common::Engine::Get()->GetGraphBatch();
  auto row = graph_batch->graphs[layer_idx]->row;
  auto device = common::Engine::Get()->GetTrainDevice();
  auto device_str = "cuda:" + std::to_string(device);

  CHECK_EQ(key, graph_batch->key);
  ::torch::Tensor tensor = ::torch::from_blob(
      row->mutable_data(), {(long long)row->shape()[0]}, [row](void* data) {},
      ::torch::TensorOptions().dtype(::torch::kI32).device(device_str));

  return tensor;
}

::torch::Tensor GetGraphCol(uint64_t key, int layer_idx) {
  auto graph_batch = common::Engine::Get()->GetGraphBatch();
  auto col = graph_batch->graphs[layer_idx]->col;
  auto device = common::Engine::Get()->GetTrainDevice();
  auto device_str = "cuda:" + std::to_string(device);

  CHECK_EQ(key, graph_batch->key);
  ::torch::Tensor tensor = ::torch::from_blob(
      col->mutable_data(), {(long long)col->shape()[0]}, [col](void* data) {},
      ::torch::TensorOptions().dtype(::torch::kI32).device(device_str));

  return tensor;
}

PYBIND11_MODULE(c_lib, m) {
  m.def("samgraph_torch_get_graph_feat", &GetGraphFeature);
  m.def("samgraph_torch_get_graph_label", &GetGraphLabel);
  m.def("samgraph_torch_get_graph_row", &GetGraphRow);
  m.def("samgraph_torch_get_graph_col", &GetGraphCol);
}

}  // namespace torch
}  // namespace samgraph
