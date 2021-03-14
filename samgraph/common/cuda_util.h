#ifndef SAMGRAPH_CUDA_UTIL_H
#define SAMGRAPH_CUDA_UTIL_H

#include <cuda_runtime.h>

#include "types.h"

namespace samgraph{
namespace common {
namespace cuda {

void Fill(float *data, size_t len, float val, cudaStream_t stream);

void PrintID(const IdType * input, const size_t num_input, cudaStream_t stream);

} // namespace cuda
} // namespace common
} // namespace samgraph

#endif // SAMGRAPH_CUDA_UTIL_H