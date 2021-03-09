#include "adapter.h"

namespace samgraph {
namespace torch {

// TorchTensor::TorchTensor(::torch::Tensor tensor) : tensor_(tensor) {}

// const DataType TorchTensor::dtype() const {
//   switch (tensor_.scalar_type()) {
//     case ::torch::kU8:
//       return DataType::kSamU8;
//     case ::torch::kI8:
//       return DataType::kSamI8;
//     // case ::torch::kShort:
//     //   return DataType::SAM_INT16;
//     case ::torch::kI32:
//       return DataType::kSamI32;
//     case ::torch::kI64:
//       return DataType::kSamI64;
//     case ::torch::kF16:
//       return DataType::kSamF16;
//     case ::torch::kF32:
//       return DataType::kSamF32;
//     case ::torch::kF64:
//       return DataType::kSamF64;
//     default:
//       throw std::logic_error("Invalid or unsupported tensor type.");
//   }
// }

// const void* TorchTensor::data() const { return tensor_.data_ptr(); }

// int64_t TorchTensor::size() const {
// #if TORCH_VERSION >= 1001000000
//   return tensor_.element_size() * tensor_.numel();
// #else
//   return tensor_.type().elementSizeInBytes() * tensor_.numel();
// #endif
// }

} // namespace torch
} // namespace samgraph
