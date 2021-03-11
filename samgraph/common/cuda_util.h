#ifndef SAMGRAPH_CUDA_UTIL_H
#define SAMGRAPH_CUDA_UTIL_H

#include <cuda_runtime.h>

namespace samgraph{
namespace common {
namespace cuda {

template<typename T>
void Fill(T *data, size_t len, T val, cudaStream_t stream);

} // namespace cuda
} // namespace common
} // namespace samgraph

#endif // SAMGRAPH_CUDA_UTIL_H