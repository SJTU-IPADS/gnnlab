#include <torch/torch.h>
#include <torch/extension.h>
#include <cusparse.h>
#include <cuda_runtime.h>

#include "../common/engine.h"
#include "../common/logging.h"
#include "ops.h"

namespace samgraph {
namespace torch {

::torch::Tensor SpmmForward(uint64_t key, unsigned int layer_idx, ::torch::Tensor input) {
    auto graph_batch = common::SamGraphEngine::GetGraphBatch();
    SAM_CHECK_EQ(key, graph_batch->key);
}

::torch::Tensor SpmmBackward(uint64_t key, ::torch::Tensor grad_output) {

}

// PYBIND11_MODULE(c_lib, m) {
//     m.def("samgraph_torch_spmm_forward", &SpmmForward);
//     m.def("samgraph_torch_spmm_backward", &SpmmBackward);
// }

} // namespace torch
} // namespace samgraph
