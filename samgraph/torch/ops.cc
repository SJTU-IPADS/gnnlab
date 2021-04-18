#include "ops.h"

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include "../common/common.h"
#include "../common/cuda/cuda_engine.h"
#include "../common/cuda/cuda_function.h"
#include "../common/logging.h"
#include "../common/profiler.h"
#include "../common/timer.h"

namespace samgraph {
namespace torch {

namespace {

bool stream_sync = true;

}

::torch::Dtype toTorchType(common::DataType dtype) {
  switch (dtype) {
    case common::kSamF32:
      return ::torch::kF32;
    case common::kSamF64:
      return ::torch::kF64;
    case common::kSamU8:
      return ::torch::kU8;
    case common::kSamI32:
      return ::torch::kI32;
    case common::kSamI8:
      return ::torch::kI8;
    case common::kSamI64:
      return ::torch::kI64;
    default:
      SAM_CHECK(0);
      return ::torch::kF32;
  }

  SAM_CHECK(0);
}

::torch::Tensor Csrmm(uint64_t key, ::torch::Tensor input) {
  auto graph_id = common::decodeGraphID(key);
  auto batch_key = common::decodeBatchKey(key);

  auto graph_batch = common::SamGraphEngine::GetEngine()->GetGraphBatch();
  auto train_graph = graph_batch->output_graph.at(graph_id);
  auto device = common::SamGraphEngine::GetEngine()->GetTrainDevice();
  auto device_str = "cuda:" + std::to_string(device);

  SAM_CHECK_EQ(batch_key, graph_batch->key);

  cusparseHandle_t handle = at::cuda::getCurrentCUDASparseHandle();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device);

  if (!train_graph->val) {
    size_t num_edge = train_graph->num_edge;
    common::Timer t0;
    auto val = common::Tensor::Empty(common::kSamF32, {num_edge}, device,
                                     "csrmm_val_" + std::to_string(key));
    double alloc_val_time = t0.Passed();

    common::Timer t1;
    common::cuda::Fill(reinterpret_cast<float*>(val->mutable_data()), num_edge,
                       1.0f, stream, stream_sync);
    train_graph->val = val;
    double fill_val_time = t1.Passed();

    common::Profiler::Get()->alloc_val_time[graph_batch->profile_idx] +=
        alloc_val_time;
    common::Profiler::Get()->fill_val_time[graph_batch->profile_idx] +=
        fill_val_time;
  }

  ::torch::Tensor output = ::torch::empty(
      {(long long)train_graph->num_row, (long long)input.sizes()[1]},
      ::torch::TensorOptions().dtype(::torch::kF32).device(device_str));
  // A is m x k
  // B is k x n
  // C is m x n
  const float alpha = 1.0f;
  int m = train_graph->num_row;     // Number of row in matrix
  int n = input.sizes()[1];         // Number of columns in input
  int k = train_graph->num_column;  // Number of column in matrix
  int nnz = train_graph->num_edge;
  const float* val_a = reinterpret_cast<const float*>(train_graph->val->data());
  const int* indptr_a =
      reinterpret_cast<const int*>(train_graph->indptr->data());
  const int* indices_a =
      reinterpret_cast<const int*>(train_graph->indices->data());
  const float* b = input.data_ptr<float>();
  int ldb = k;
  const float beta = 0.0f;
  int ldc = m;
  float* c = output.data_ptr<float>();

  SAM_CHECK_EQ(k, input.sizes()[0]);
  SAM_CHECK_EQ(m + 1, train_graph->indptr->shape()[0]);
  SAM_CHECK_EQ(nnz, train_graph->indices->shape()[0]);

  common::Timer t2;
  cusparseMatDescr_t descr_a;
  CUSPARSE_CALL(cusparseCreateMatDescr(&descr_a));
  CUSPARSE_CALL(cusparseSetMatType(descr_a, CUSPARSE_MATRIX_TYPE_GENERAL));
  CUSPARSE_CALL(cusparseSetMatIndexBase(descr_a, CUSPARSE_INDEX_BASE_ZERO));

  CUSPARSE_CALL(cusparseScsrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n,
                               k, nnz, &alpha, descr_a, val_a, indptr_a,
                               indices_a, b, ldb, &beta, c, ldc));

  CUSPARSE_CALL(cusparseDestroyMatDescr(descr_a));
  if (stream_sync) {
    CUDA_CALL(cudaStreamSynchronize(stream));
  }

  double csrmm_time = t2.Passed();
  common::Profiler::Get()->csrmm_time[graph_batch->profile_idx] += csrmm_time;

  return output;
}

::torch::Tensor CsrmmTranspose(uint64_t key, ::torch::Tensor input) {
  auto graph_id = common::decodeGraphID(key);
  auto batch_key = common::decodeBatchKey(key);

  auto graph_batch = common::SamGraphEngine::GetEngine()->GetGraphBatch();
  auto train_graph = graph_batch->output_graph.at(graph_id);
  auto device = common::SamGraphEngine::GetEngine()->GetTrainDevice();
  auto device_str = "cuda:" + std::to_string(device);

  SAM_CHECK_EQ(batch_key, graph_batch->key);

  cusparseHandle_t handle = at::cuda::getCurrentCUDASparseHandle();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device);

  if (!train_graph->val) {
    size_t num_edge = train_graph->num_edge;
    auto val = common::Tensor::Empty(common::kSamF32, {num_edge}, device,
                                     "csrmmsp_val_" + std::to_string(key));
    common::cuda::Fill(reinterpret_cast<float*>(val->mutable_data()), num_edge,
                       1.0f, stream, stream_sync);
    train_graph->val = val;
  }

  ::torch::Tensor output = ::torch::empty(
      {(long long)train_graph->num_column, (long long)input.sizes()[1]},
      ::torch::TensorOptions().dtype(::torch::kF32).device(device_str));

  // A is m x k, T(A) is k x m
  // B is m x n
  // C is k x n
  const float alpha = 1.0f;
  int m = train_graph->num_row;     // Number of row in matrix
  int n = input.sizes()[1];         // Number of columns in input
  int k = train_graph->num_column;  // Number of column in matrix
  int nnz = train_graph->num_edge;
  const float* val_a = reinterpret_cast<const float*>(train_graph->val->data());
  const int* indptr_a =
      reinterpret_cast<const int*>(train_graph->indptr->data());
  const int* indices_a =
      reinterpret_cast<const int*>(train_graph->indices->data());
  const float* b = input.data_ptr<float>();
  int ldb = m;  // Leading dimension of b has changed from k to m
  const float beta = 0.0f;
  int ldc = k;  // Leading dimension of c has changed from m to k
  float* c = output.data_ptr<float>();

  common::Timer t0;
  SAM_CHECK_EQ(m, input.sizes()[0]);
  SAM_CHECK_EQ(m + 1, train_graph->indptr->shape()[0]);
  SAM_CHECK_EQ(nnz, train_graph->indices->shape()[0]);

  cusparseMatDescr_t descr_a;
  CUSPARSE_CALL(cusparseCreateMatDescr(&descr_a));
  CUSPARSE_CALL(cusparseSetMatType(descr_a, CUSPARSE_MATRIX_TYPE_GENERAL));
  CUSPARSE_CALL(cusparseSetMatIndexBase(descr_a, CUSPARSE_INDEX_BASE_ZERO));

  CUSPARSE_CALL(cusparseScsrmm(handle, CUSPARSE_OPERATION_TRANSPOSE, m, n, k,
                               nnz, &alpha, descr_a, val_a, indptr_a, indices_a,
                               b, ldb, &beta, c, ldc));

  CUSPARSE_CALL(cusparseDestroyMatDescr(descr_a));
  if (stream_sync) {
    CUDA_CALL(cudaStreamSynchronize(stream));
  }

  double csrmm_transpose_time = t0.Passed();
  common::Profiler::Get()->csrmm_transpose_time[graph_batch->profile_idx] +=
      csrmm_transpose_time;

  return output;
}

::torch::Tensor GetGraphFeature(uint64_t key) {
  auto graph_batch = common::SamGraphEngine::GetEngine()->GetGraphBatch();
  auto feat = graph_batch->input_feat;
  auto device = common::SamGraphEngine::GetEngine()->GetTrainDevice();
  auto device_str = "cuda:" + std::to_string(device);

  SAM_CHECK_EQ(key, graph_batch->key);
  SAM_LOG(DEBUG) << "GetGraphFeature feat shape " << feat->shape().size()
                 << " dim0: " << feat->shape()[0]
                 << " dim1: " << feat->shape()[1] << "device_str "
                 << device_str;

  ::torch::Tensor tensor = ::torch::from_blob(
      feat->mutable_data(),
      {(long long)feat->shape()[0], (long long)feat->shape()[1]},
      [feat](void* data) {},
      ::torch::TensorOptions().dtype(::torch::kF32).device(device_str));

  return tensor;
}

::torch::Tensor GetGraphLabel(uint64_t key) {
  auto graph_batch = common::SamGraphEngine::GetEngine()->GetGraphBatch();
  auto label = graph_batch->output_label;
  auto device = common::SamGraphEngine::GetEngine()->GetTrainDevice();
  auto device_str = "cuda:" + std::to_string(device);

  SAM_CHECK_EQ(key, graph_batch->key);
  ::torch::Tensor tensor = ::torch::from_blob(
      label->mutable_data(), {(long long)label->shape()[0]},
      [label](void* data) {},
      ::torch::TensorOptions().dtype(::torch::kI64).device(device_str));

  return tensor;
}

::torch::Tensor GetGraphRow(uint64_t key, int layer_idx) {
  auto graph_batch = common::SamGraphEngine::GetEngine()->GetGraphBatch();
  auto row = graph_batch->output_graph[layer_idx]->row;
  auto device = common::SamGraphEngine::GetEngine()->GetTrainDevice();
  auto device_str = "cuda:" + std::to_string(device);

  SAM_CHECK_EQ(key, graph_batch->key);
  ::torch::Tensor tensor = ::torch::from_blob(
      row->mutable_data(), {(long long)row->shape()[0]}, [row](void* data) {},
      ::torch::TensorOptions().dtype(::torch::kI32).device(device_str));

  return tensor;
}

::torch::Tensor GetGraphCol(uint64_t key, int layer_idx) {
  auto graph_batch = common::SamGraphEngine::GetEngine()->GetGraphBatch();
  auto col = graph_batch->output_graph[layer_idx]->col;
  auto device = common::SamGraphEngine::GetEngine()->GetTrainDevice();
  auto device_str = "cuda:" + std::to_string(device);

  SAM_CHECK_EQ(key, graph_batch->key);
  ::torch::Tensor tensor = ::torch::from_blob(
      col->mutable_data(), {(long long)col->shape()[0]}, [col](void* data) {},
      ::torch::TensorOptions().dtype(::torch::kI32).device(device_str));

  return tensor;
}

PYBIND11_MODULE(c_lib, m) {
  m.def("samgraph_torch_get_graph_feat", &GetGraphFeature);
  m.def("samgraph_torch_get_graph_label", &GetGraphLabel);
  m.def("samgraph_torch_csrmm", &Csrmm);
  m.def("samgraph_torch_csrmm_transpose", &CsrmmTranspose);
  m.def("samgraph_torch_get_graph_row", &GetGraphRow);
  m.def("samgraph_torch_get_graph_col", &GetGraphCol);
}

}  // namespace torch
}  // namespace samgraph
