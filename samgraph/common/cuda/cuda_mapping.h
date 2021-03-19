#ifndef SAMGRAPH_CUDA_MAPPING_H
#define SAMGRAPH_CUDA_MMAPING_H

#include <cuda_runtime.h>

#include "../types.h"
#include "cuda_hashtable.h"

namespace samgraph {
namespace common {
namespace cuda {

void MapEdges(const IdType * const global_src,
              IdType * const new_global_src,
              const IdType * const global_dst,
              IdType * const new_global_dst,
              const size_t num_edges,
              DeviceOrderedHashTable mapping,
              cudaStream_t stream);

} // namespace cuda
} // namespace common
} // namespace samgraph

#endif // SAMGRAPH_CUDA_MMAPING_H