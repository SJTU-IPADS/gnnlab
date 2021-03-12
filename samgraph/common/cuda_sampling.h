#ifndef SAMGRAPH_CUDA_SAMPLER_H
#define SAMGRAPH_CUDA_SAMPLER_H

#include <cuda_runtime.h>

#include "types.h"

namespace samgraph{
namespace common {
namespace cuda {

void DeviceSample(const IdType *indptr, const IdType *indices, const IdType *input, const size_t num_input,
                  const size_t fanout, IdType *out_src, IdType *out_dst, size_t *num_out, cudaStream_t stream);

} // namespace cuda
} // namespace common
} // namespace samgraph


#endif // SAMGRAPH_CUDA_SAMPLER_H