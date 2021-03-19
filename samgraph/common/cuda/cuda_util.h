#ifndef SAMGRAPH_CUDA_UTIL_H
#define SAMGRAPH_CUDA_UTIL_H

#include <cuda_runtime.h>

#include "../types.h"

namespace samgraph{
namespace common {
namespace cuda {

void Fill(float *data, size_t len, float val, cudaStream_t stream, bool sync=false);

void PrintID(const IdType * input, const size_t num_input, cudaStream_t stream);

void AssertVal(const int64_t *input, const size_t num_input, const int64_t min, const int64_t max, cudaStream_t stream);

} // namespace cuda
} // namespace common
} // namespace samgraph

#endif // SAMGRAPH_CUDA_UTIL_H