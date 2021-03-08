#include <cassert>

#include "cuda_mapping.h"
#include "cuda_hashtable.h"

namespace {

using samgraph::common::cuda;

template <int BLOCK_SIZE, size_t TILE_SIZE>
__device__ void map_node_ids (
    const nodeid_t *const global,
    nodeid_t * const new_global,
    const size_t num_nodes,
    const DeviceOrderedHashTable &table) {
    assert(BLOCK_SIZE == blockDim.x);
    
    using Mapping = typename OrderedHashTable::Mapping;

    const size_t tile_start = TILE_SIZE * blockIdx.x;
    const size_t tile_end = min(TILE_SIZE * (blockIdx.x + 1), num_nodes);

    for (size_t idx = threadIdx.x + tile_start; idx < tile_end; idx += BLOCK_SIZE) {
        const Mapping &mapping = *table.Search(global[idx]);
        new_global[idx] = mapping.local;
    }
}

template <int BLOCK_SIZE, size_t TILE_SIZE>
__global__ void map_edge_ids(
    const nodeid_t * const global_src,
    nodeid_t * const new_global_src,
    const nodeid_t * const global_dst,
    nodeid_t * const new_global_dst,
    const size_t num_edges,
    DeviceOrderedHashTable &mapping
) {
    assert(BLOCK_SIZE == blockDim.x);

    if (blockIdx.y == 0) {
        map_node_ids<BLOCK_SIZE, TILE_SIZE>(
            global_src,
            new_global_src,
            num_edges,
            mapping
        )
    } else {
        map_node_ids<BLOCK_SIZE, TILE_SIZE>(
            global_dst,
            new_global_dst,
            num_edges,
            mapping
        )
    }
}

inline size_t RoundUpDiv(
    const size_t num,
    const size_t divisor) {
  return num / divisor + (num % divisor == 0 ? 0 : 1);
}

} // namespace

namespace samgraph {
namespace common {
namespace cuda {

void MapEdges(const nodeid_t * const global_src,
              nodeid_t * const new_global_src,
              const nodeid_t * const global_dst,
              nodeid_t * const new_global_dst,
              const size_t num_edges,
              DeviceOrderedHashTable mapping,
              cudaStream_t stream) {
    
    const dim3 grid(RoundUpDiv(num_edges, kCudaTileSize), 2);
    const dim3 block(kCudaBlockSize);
    
    map_edge_ids<kCudaBlockSize, kCudaTileSize><<<
        grid,
        block,
        0,
        stream>>>(
        global_src,
        new_global_src,
        global_dst,
        new_gloval_dst,
        num_edges,
        mapping);
}

} // namespace cuda
} // namespace common
} // namespace samgraph
