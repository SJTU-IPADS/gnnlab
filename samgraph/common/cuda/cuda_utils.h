#ifndef SAMGRAPH_CUDA_UTILS_H
#define SAMGRAPH_CUDA_UTILS_H

#include <cuda_runtime.h>

namespace samgraph {
namespace common {
namespace cuda {

/**
 * This structure is used with cub's block-level prefixscan in order to
 * keep a running sum as items are iteratively processed.
 */
template <typename T>
struct BlockPrefixCallbackOp {
  T _running_total;

  __device__ BlockPrefixCallbackOp(const T running_total)
      : _running_total(running_total) {}

  __device__ T operator()(const T block_aggregate) {
    const T old_prefix = _running_total;
    _running_total += block_aggregate;
    return old_prefix;
  }
};

}  // namespace cuda
}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_CUDA_UTILS_H