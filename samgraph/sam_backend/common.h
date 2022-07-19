#pragma once
#include "../common/common.h"
#include "../common/logging.h"
#include "../common/profiler.h"
#include "../common/timer.h"
#include "constants.h"
#include "../common/cuda/cuda_device.h"
#include <cublas_v2.h>
#include <cudnn.h>
#include <curand.h>

#define CUDNN_CALL(status)                           \
  {                                                  \
    CHECK((status) == CUDNN_STATUS_SUCCESS)          \
        << "CUDNN: " << cudnnGetErrorString(status); \
  }

// #if defined(CUBLAS_VERSION) && CUBLAS_VERSION < 11400
#if true
// in cuda <= 11.3, there is no string for cublas status...
#define CUBLAS_CALL(status)                           \
  {                                                  \
    CHECK((status) == CUBLAS_STATUS_SUCCESS)          \
        << "CUBLAS ERROR "; \
  }
#else
#define CUBLAS_CALL(status)                           \
  {                                                  \
    CHECK((status) == CUBLAS_STATUS_SUCCESS)          \
        << "CUBLAS : " << cublasGetStatusString(status); \
  }
#endif

#define CURAND_CALL(status)                                      \
  {                                                          \
    curandStatus_t e = (status);                                  \
    CHECK(e == CURAND_STATUS_SUCCESS) \
        << "CURAND ERROR";                \
  }

namespace samgraph {
namespace sam_backend {

using common::Context;
using common::StreamHandle;
using common::Tensor;
using common::TensorPtr;
using common::RoundUpDiv;
using common::Constant;
using common::IdType;
using common::Id64Type;
using common::cuda::GPUDevice;
using common::Device;
using common::Task;
using common::TaskPtr;
using common::TrainGraph;
using common::Timer;
using common::Profiler;
using common::LogStepItem;
using TrainGraphPtr = std::shared_ptr<TrainGraph>;

using TrainType = float;
// typedef float TrainType;

template <typename TT>
inline cudaDataType to_cuda_data_type() { CHECK(false); return CUDA_R_8U; }
template <>
inline cudaDataType to_cuda_data_type<float>() { return CUDA_R_32F; }
template <>
inline cudaDataType to_cuda_data_type<double>() { return CUDA_R_64F; }
template <>
inline cudaDataType to_cuda_data_type<uint8_t>() { return CUDA_R_8U; }
template <>
inline cudaDataType to_cuda_data_type<int32_t>() { return CUDA_R_32I; }
template <>
inline cudaDataType to_cuda_data_type<uint32_t>() { return CUDA_R_32U; }
template <>
inline cudaDataType to_cuda_data_type<uint64_t>() { return CUDA_R_64U; }
template <>
inline cudaDataType to_cuda_data_type<Id64Type>() { return CUDA_R_64U; }

template <typename TT>
inline cudnnDataType_t to_cudnn_data_type() { CHECK(false); return CUDNN_DATA_UINT8; }
template <>
inline cudnnDataType_t to_cudnn_data_type<float>() { return CUDNN_DATA_FLOAT; }
template <>
inline cudnnDataType_t to_cudnn_data_type<double>() { return CUDNN_DATA_DOUBLE; }
template <>
inline cudnnDataType_t to_cudnn_data_type<uint8_t>() { return CUDNN_DATA_UINT8; }
template <>
inline cudnnDataType_t to_cudnn_data_type<int32_t>() { return CUDNN_DATA_INT32; }
template <>
inline cudnnDataType_t to_cudnn_data_type<uint32_t>() { return CUDNN_DATA_INT32; }
template <>
inline cudnnDataType_t to_cudnn_data_type<uint64_t>() { return CUDNN_DATA_INT64; }
template <>
inline cudnnDataType_t to_cudnn_data_type<Id64Type>() { return CUDNN_DATA_INT64; }

template <typename TT>
inline common::DataType to_data_type() { CHECK(false); return common::kU8; }
template <>
inline common::DataType to_data_type<float>() { return common::kF32; }
template <>
inline common::DataType to_data_type<double>() { return common::kF64; }
template <>
inline common::DataType to_data_type<uint8_t>() { return common::kU8; }
template <>
inline common::DataType to_data_type<int32_t>() { return common::kI32; }
template <>
inline common::DataType to_data_type<uint32_t>() { return common::kI32; }
template <>
inline common::DataType to_data_type<uint64_t>() { return common::kI64; }
template<>
inline common::DataType to_data_type<Id64Type>() { return common::kI64; }

namespace {
template <typename TT>
inline void check_type(const TensorPtr &ptr) {
  CHECK(ptr->Type() == to_data_type<TT>());
}
} // namespace

template <typename T>
T *tensor_cast(const TensorPtr &ptr) {
  check_type<T>(ptr);
  return static_cast<T *>(ptr->MutableData());
}
template <typename T>
const T *tensor_cast_const(const TensorPtr &ptr) {
  check_type<T>(ptr);
  return static_cast<const T *>(ptr->Data());
}

class GradTensor;
using GradTensorPtr = std::shared_ptr<GradTensor>;
class GradTensor {
 public:
  GradTensor(bool require_grad = false);
  GradTensor(TensorPtr d, bool require_grad = false);
  GradTensor(common::DataType, std::vector<size_t> shape, Context ctx, std::string name, bool require_grad = false);
  inline void ChangeData(TensorPtr d) { _data = d; CHECK(_require_grad == false); CHECK(_grad->Defined() == false); }
  inline bool RequireGrad() const { return _require_grad; }
  inline TensorPtr data() const { return _data; }
  inline TensorPtr grad() const { return _grad; }
  inline common::DataType Type() const { return _data->Type(); }
  inline const std::vector<size_t>& Shape() const { return _data->Shape(); }
  inline size_t NumBytes() const { return _data->NumBytes(); }
  inline Context Ctx() const { return _data->Ctx(); }
  inline size_t AddConsumer() { return _num_consumer++; }
  inline size_t GetConsumer() { return _num_consumer; }
  void Resize(common::DataType, std::vector<size_t> shape, Context ctx, std::string name);
  void Print() const;

  static GradTensorPtr Null(bool require_grad = false);
  static GradTensorPtr FromTensor(TensorPtr d, bool require_grad = false);
  static GradTensorPtr Empty(common::DataType, std::vector<size_t> shape, Context ctx, std::string name, bool require_grad = false);

 private:
  TensorPtr _data, _grad;
  size_t _data_actual_size = 0, _grad_actual_size = 0;
  const bool _require_grad = false;
  size_t _num_consumer = 0;
};

void PrintTensor(TensorPtr ptr, std::string prefix="");

} // namespace sam_backend
} // namespace samgraph