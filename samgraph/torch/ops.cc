#include <torch/torch.h>
#include <torch/extension.h>
#include <cusparse.h>
#include <cuda_runtime.h>

#include "ops.h"
#include "../common/engine.h"
#include "../common/logging.h"
#include "../common/cuda_util.h"
#include "../common/common.h"

namespace samgraph {
namespace torch {

::torch::Dtype toTorchType(common::DataType dtype) {
    switch(dtype) {
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

    auto graph_batch = common::SamGraphEngine::GetGraphBatch();
    auto train_graph = graph_batch->output_graph.at(graph_id);
    auto device = common::SamGraphEngine::GetTrainDevice();
    auto device_str = "cuda:" + std::to_string(device);

    SAM_CHECK_EQ(key, graph_batch->key);

    cusparseHandle_t handle;
    cusparseMatDescr_t descr_a;
    cudaStream_t stream;

    CUDA_CALL(cudaSetDevice(device));
    stream = *common::SamGraphEngine::GetTrainStream();

    if (!train_graph->val) {
        size_t num_edge = train_graph->num_edge;
        auto val = common::Tensor::Empty(common::kSamF32, {num_edge}, device, "csrmm_val_" + std::to_string(key));
        common::cuda::Fill(reinterpret_cast<float*>(val->mutable_data()),
                           num_edge, 1.0f, stream);
        train_graph->val = val;
    }

    ::torch::Tensor output = ::torch::empty({(long long)train_graph->num_row,
                                             (long long)input.sizes()[1]},
                                             ::torch::TensorOptions().dtype(::torch::kF32)
                                                                     .device(device_str));
    // A is m x k
    // B is k x n
    // C is m x n
    const float alpha = 1.0f;
    int m = train_graph->num_row; // Number of row in matrix
    int n = input.sizes()[1]; // Number of columns in input
    int k = train_graph->num_column; // Number of column in matrix
    int nnz = train_graph->num_edge;
    const float* val_a = reinterpret_cast<const float*>(train_graph->val->data());
    const int* indptr_a = reinterpret_cast<const int*>(train_graph->indptr->data());
    const int* indices_a = reinterpret_cast<const int*>(train_graph->indices->data());
    const float *b = input.data_ptr<float>();
    int ldb = k;
    const float beta = 0.0f;
    int ldc = m;
    float *c = output.data_ptr<float>();

    CUDA_CALL(cudaMalloc((void **)&c, m * n *sizeof(float)));

    CUSPARSE_CALL(cusparseCreate(&handle));
    CUSPARSE_CALL(cusparseSetStream(handle, stream));
    
    CUSPARSE_CALL(cusparseCreateMatDescr(&descr_a));
    CUSPARSE_CALL(cusparseSetMatType(descr_a, CUSPARSE_MATRIX_TYPE_GENERAL));
    CUSPARSE_CALL(cusparseSetMatIndexBase(descr_a, CUSPARSE_INDEX_BASE_ZERO));

    CUSPARSE_CALL(cusparseScsrmm(handle,
                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                  m, n, k, nnz, &alpha,
                  descr_a,
                  val_a,
                  indptr_a,
                  indices_a,
                  b,
                  ldb,
                  &beta,
                  c,
                  ldc));

    CUDA_CALL(cudaStreamSynchronize(stream));
    CUSPARSE_CALL(cusparseDestroyMatDescr(descr_a));
    CUSPARSE_CALL(cusparseDestroy(handle));

    return output;
}

::torch::Tensor CsrmmTranspose(uint64_t key, ::torch::Tensor input) {
    auto graph_id = common::decodeGraphID(key);

    auto graph_batch = common::SamGraphEngine::GetGraphBatch();
    auto train_graph = graph_batch->output_graph.at(graph_id);
    auto device = common::SamGraphEngine::GetTrainDevice();
    auto device_str = "cuda:" + std::to_string(device);

    SAM_CHECK_EQ(key, graph_batch->key);

    cusparseHandle_t handle;
    cusparseMatDescr_t descr_a;
    cudaStream_t stream;

    CUDA_CALL(cudaSetDevice(device));
    stream = *common::SamGraphEngine::GetTrainStream();

    if (!train_graph->val) {
        size_t num_edge = train_graph->num_edge;
        auto val = common::Tensor::Empty(common::kSamF32, {num_edge}, device, "csrmmsp_val_" + std::to_string(key));
        common::cuda::Fill(reinterpret_cast<float*>(val->mutable_data()),
                           num_edge, 1.0f, stream);
        train_graph->val = val;
    }

    ::torch::Tensor output = ::torch::empty({(long long)train_graph->num_column,
                                             (long long)input.sizes()[1]},
                                             ::torch::TensorOptions().dtype(::torch::kF32)
                                                                     .device(device_str));

    // A is m x k, T(A) is k x m
    // B is m x n
    // C is k x n
    const float alpha = 1.0f;
    int m = train_graph->num_row; // Number of row in matrix
    int n = input.sizes()[1]; // Number of columns in input
    int k = train_graph->num_column; // Number of column in matrix
    int nnz = train_graph->num_edge;
    const float* val_a = reinterpret_cast<const float*>(train_graph->val->data());
    const int* indptr_a = reinterpret_cast<const int*>(train_graph->indptr->data());
    const int* indices_a = reinterpret_cast<const int*>(train_graph->indices->data());
    const float *b = input.data_ptr<float>();
    int ldb = m; // Leading dimension of b has changed from m to k
    const float beta = 0.0f;
    int ldc = k; // Leading dimension of c has changed from k to m
    float *c = output.data_ptr<float>();

    CUDA_CALL(cudaMalloc((void **)&c, k * n *sizeof(float)));

    CUSPARSE_CALL(cusparseCreate(&handle));
    CUSPARSE_CALL(cusparseSetStream(handle, stream));
    
    CUSPARSE_CALL(cusparseCreateMatDescr(&descr_a));
    CUSPARSE_CALL(cusparseSetMatType(descr_a, CUSPARSE_MATRIX_TYPE_GENERAL));
    CUSPARSE_CALL(cusparseSetMatIndexBase(descr_a, CUSPARSE_INDEX_BASE_ZERO));

    CUSPARSE_CALL(cusparseScsrmm(handle,
                  CUSPARSE_OPERATION_TRANSPOSE,
                  m, n, k, nnz, &alpha,
                  descr_a,
                  val_a,
                  indptr_a,
                  indices_a,
                  b,
                  ldb,
                  &beta,
                  c,
                  ldc));

    CUDA_CALL(cudaStreamSynchronize(stream));
    CUSPARSE_CALL(cusparseDestroyMatDescr(descr_a));
    CUSPARSE_CALL(cusparseDestroy(handle));

    return output;
}

::torch::Tensor GetGraphFeature(uint64_t key) {
    auto graph_batch = common::SamGraphEngine::GetGraphBatch();
    auto feat = graph_batch->input_feat;
    auto device = common::SamGraphEngine::GetTrainDevice();
    auto device_str = "cuda:" + std::to_string(device);
    
    SAM_CHECK_EQ(key, graph_batch->key);
    SAM_LOG(DEBUG) << "GetGraphFeature feat shape " << feat->shape().size() << " dim0: " << feat->shape()[0] << " dim1: " << feat->shape()[1]
                   << "device_str " << device_str;

    ::torch::Tensor tensor = ::torch::from_blob(
        feat->mutable_data(),
        {(long long)feat->shape()[0], (long long)feat->shape()[1]},
        [key] (void *data) {
            SAM_LOG(DEBUG) << "Torch feature tensor with key " << key << " has been freed";
            CUDA_CALL(cudaFree(data));
        },
        ::torch::TensorOptions().dtype(::torch::kF32)
                                .device(device_str)
    );

    feat->ConsumeData();
    SAM_LOG(DEBUG) << "GetGraphFeature: Consume data";
    return tensor;
}

::torch::Tensor GetGraphLabel(uint64_t key) {
    auto graph_batch = common::SamGraphEngine::GetGraphBatch();
    auto label = graph_batch->output_label;
    auto device = common::SamGraphEngine::GetTrainDevice();
    auto device_str = "cuda:" + std::to_string(device);

    SAM_CHECK_EQ(key, graph_batch->key);
    ::torch::Tensor tensor = ::torch::from_blob(
        label->mutable_data(),
        {(long long)label->shape()[0]},
        [key] (void *data) {
            SAM_LOG(DEBUG) << "Torch label tensor with key " << key << " has been freed";
            CUDA_CALL(cudaFree(data));
        },
        ::torch::TensorOptions().dtype(::torch::kI64)
                                .device(device_str)
    );

    label->ConsumeData();
    SAM_LOG(DEBUG) << "GetGraphLabel: label ConsumeData";
    return tensor;
}

::torch::Tensor CudaTensor() {
    void *ptr;
    CUDA_CALL(cudaSetDevice(0));
    CUDA_CALL(cudaMalloc(&ptr, 20ul));
    SAM_LOG(DEBUG) << "CudaTensor: original ptr addr " << ptr;
    ::torch::Tensor tensor = ::torch::from_blob(
        ptr,
        {5ll},
        [](void *data) {
            CUDA_CALL(cudaFree(data));
        },
        ::torch::TensorOptions().dtype(::torch::kF32)
                                .device("cuda:0")
    );
    SAM_LOG(DEBUG) << "CudaTensor: torchTensor ptr addr " << tensor.data_ptr();
    return tensor;
}

PYBIND11_MODULE(c_lib, m) {
    m.def("samgraph_torch_get_graph_feat", &GetGraphFeature);
    m.def("samgraph_torch_get_graph_label", &GetGraphLabel);
    m.def("samgraph_torch_csrmm", &Csrmm);
    m.def("samgraph_torch_csrmm_tranpose", &CsrmmTranspose);
    m.def("samgraph_torch_cuda_tensor", &CudaTensor);
}

} // namespace torch
} // namespace samgraph
