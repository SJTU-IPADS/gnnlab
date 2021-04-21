#include "adapter.h"

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
      ::torch::TensorOptions().dtype(::torch::kF32).device(device));

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
      ::torch::TensorOptions().dtype(::torch::kI64).device(device));

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

PYBIND11_MODULE(c_lib, m) {
  m.def("samgraph_torch_get_graph_feat", &GetGraphFeature);
  m.def("samgraph_torch_get_graph_label", &GetGraphLabel);
  m.def("samgraph_torch_get_graph_row", &GetGraphRow);
  m.def("samgraph_torch_get_graph_col", &GetGraphCol);
}

}  // namespace torch
}  // namespace samgraph
