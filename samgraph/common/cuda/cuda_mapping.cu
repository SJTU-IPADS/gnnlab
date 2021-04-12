#include <cassert>
#include <cstdio>

#include "../config.h"
#include "cuda_function.h"
#include "cuda_hashtable.h"

namespace samgraph {
namespace common {
namespace cuda {

template <int BLOCK_SIZE, size_t TILE_SIZE>
__device__ void map_node_ids(const IdType *const global,
                             IdType *const new_global, const size_t num_input,
                             const DeviceOrderedHashTable &table) {
  assert(BLOCK_SIZE == blockDim.x);

  using Bucket = typename OrderedHashTable::Bucket;

  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = min(TILE_SIZE * (blockIdx.x + 1), num_input);

  for (size_t idx = threadIdx.x + block_start; idx < block_end;
       idx += BLOCK_SIZE) {
    const Bucket &bucket = *table.Search(global[idx]);
    new_global[idx] = bucket.local;
  }
}

template <int BLOCK_SIZE, size_t TILE_SIZE>
__global__ void
map_edge_ids(const IdType *const global_src, IdType *const new_global_src,
             const IdType *const global_dst, IdType *const new_global_dst,
             const size_t num_edges, DeviceOrderedHashTable table) {
  assert(BLOCK_SIZE == blockDim.x);
  assert(2 == gridDim.y);

  if (blockIdx.y == 0) {
    map_node_ids<BLOCK_SIZE, TILE_SIZE>(global_src, new_global_src, num_edges,
                                        table);
  } else {
    map_node_ids<BLOCK_SIZE, TILE_SIZE>(global_dst, new_global_dst, num_edges,
                                        table);
  }
}

void MapEdges(const IdType *const global_src, IdType *const new_global_src,
              const IdType *const global_dst, IdType *const new_global_dst,
              const size_t num_edges, DeviceOrderedHashTable table,
              cudaStream_t stream) {
  const size_t num_tiles =
      (num_edges + Config::kCudaTileSize - 1) / Config::kCudaTileSize;
  const dim3 grid(num_tiles, 2);
  const dim3 block(Config::kCudaBlockSize);

  map_edge_ids<Config::kCudaBlockSize, Config::kCudaTileSize>
      <<<grid, block, 0, stream>>>(global_src, new_global_src, global_dst,
                                   new_global_dst, num_edges, table);
  CUDA_CALL(cudaStreamSynchronize(stream));
}

} // namespace cuda
} // namespace common
} // namespace samgraph
