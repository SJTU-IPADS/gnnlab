#ifndef SAMGRAPH_TORCH_ADAPTER_H
#define SAMGRAPH_TORCH_ADAPTER_H

#include <torch/torch.h>

#include "../common/common.h"

namespace samgraph {
namespace torch {

using namespace samgraph::common;

// class TorchTensor : public Tensor {
//  public:
//   TorchTensor(::torch::Tensor tensor);
//   virtual const DataType dtype() const override;
//   virtual const TensorShape shape() const override;
//   virtual const void* data() const override;
//   virtual int64_t size() const override;

//  protected:
//   ::torch::Tensor tensor_;
// }

} // namespace torch
} // namespace samgraph

#endif // SAMGRAPH_TORCH_ADAPTER_H