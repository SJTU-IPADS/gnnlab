#include <cassert>

#include "config.h"
#include "cuda_mapping.h"
#include "cuda_hashtable.h"

namespace {

inline size_t RoundUpDiv(
    const size_t num,
    const size_t divisor) {
  return num / divisor + (num % divisor == 0 ? 0 : 1);
}

} // namespace

namespace samgraph {
namespace common {
namespace cuda {

template <int BLOCK_SIZE, size_t TILE_SIZE>
__device__ void map_node_ids (
    const IdType *const global,
    IdType * const new_global,
    const size_t num_nodes,
    const DeviceOrderedHashTable &table) {
    assert(BLOCK_SIZE == blockDim.x);
    
    using Bucket = typename OrderedHashTable::Bucket;

    const size_t tile_start = TILE_SIZE * blockIdx.x;
    const size_t tile_end = min(TILE_SIZE * (blockIdx.x + 1), num_nodes);

    for (size_t idx = threadIdx.x + tile_start; idx < tile_end; idx += BLOCK_SIZE) {
        const Bucket &bucket = *table.Search(global[idx]);
        new_global[idx] = bucket.local;
    }
}

template <int BLOCK_SIZE, size_t TILE_SIZE>
__global__ void map_edge_ids(
    const IdType * const global_src,
    IdType * const new_global_src,
    const IdType * const global_dst,
    IdType * const new_global_dst,
    const size_t num_edges,
    DeviceOrderedHashTable &bucket
) {
    assert(BLOCK_SIZE == blockDim.x);

    if (blockIdx.y == 0) {
        map_node_ids<BLOCK_SIZE, TILE_SIZE>(
            global_src,
            new_global_src,
            num_edges,
            bucket
        );
    } else {
        map_node_ids<BLOCK_SIZE, TILE_SIZE>(
            global_dst,
            new_global_dst,
            num_edges,
            bucket
        );
    }
}

void MapEdges(const IdType * const global_src,
              IdType * const new_global_src,
              const IdType * const global_dst,
              IdType * const new_global_dst,
              const size_t num_edges,
              DeviceOrderedHashTable bucket,
              cudaStream_t stream) {
    
    const dim3 grid(RoundUpDiv(num_edges, Config::kCudaTileSize), 2);
    const dim3 block(Config::kCudaBlockSize);
    
    map_edge_ids<Config::kCudaBlockSize, Config::kCudaTileSize><<<
        grid,
        block,
        0,
        stream>>>(
        global_src,
        new_global_src,
        global_dst,
        new_global_dst,
        num_edges,
        bucket);
}

} // namespace cuda
} // namespace common
} // namespace samgraph
