#include "adapter.h"

namespace samgraph {
namespace torch {

TorchTensor::TorchTensor(::torch::Tensor tensor) : tensor_(tensor) {}

const DataType TorchTensor::dtype() const {
  switch (tensor_.scalar_type()) {
    case ::torch::kU8:
      return DataType::SAM_UINT8;
    case ::torch::kI8:
      return DataType::SAM_INT8;
    // case ::torch::kShort:
    //   return DataType::SAM_INT16;
    case ::torch::kI32:
      return DataType::SAM_INT32;
    case ::torch::kI64:
      return DataType::SAM_INT64;
    case ::torch::kF16:
      return DataType::SAM_FLOAT16;
    case ::torch::kF32:
      return DataType::SAM_FLOAT32;
    case ::torch::kF64:
      return DataType::SAM_FLOAT64;
    default:
      throw std::logic_error("Invalid or unsupported tensor type.");
  }
}

const TensorShape TorchTensor::shape() const {
  TensorShape shape;
  for (int idx = 0; idx < tensor_.dim(); ++idx) {
    shape.AddDim(tensor_.size(idx));
  }
  return shape;
}

const void* TorchTensor::data() const { return tensor_.data_ptr(); }

int64_t TorchTensor::size() const {
#if TORCH_VERSION >= 1001000000
  return tensor_.element_size() * tensor_.numel();
#else
  return tensor_.type().elementSizeInBytes() * tensor_.numel();
#endif
}

} // namespace torch
} // namespace samgraph
