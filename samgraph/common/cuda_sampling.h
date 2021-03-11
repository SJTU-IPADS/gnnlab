#ifndef SAMGRAPH_CUDA_SAMPLER_H
#define SAMGRAPH_CUDA_SAMPLER_H

#include <cuda_runtime.h>

#include "types.h"

namespace samgraph{
namespace common {
namespace cuda {

void DeviceSample(const nodeid_t *indptr, const nodeid_t *indices, const size_t num_node, const size_t num_edge, const nodeid_t *input,
                  const size_t num_input, const size_t fanout, nodeid_t *out_src, nodeid_t *out_dst, size_t *num_out, cudaStream_t stream);

} // namespace cuda
} // namespace common
} // namespace samgraph


#endif // SAMGRAPH_CUDA_SAMPLER_H