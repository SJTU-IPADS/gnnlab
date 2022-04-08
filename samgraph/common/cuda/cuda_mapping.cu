/*
 * Copyright 2022 Institute of Parallel and Distributed Systems, Shanghai Jiao Tong University
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include <cassert>
#include <cstdio>

#include "../common.h"
#include "../constant.h"
#include "../device.h"
#include "cuda_function.h"
#include "cuda_hashtable.h"

namespace samgraph {
namespace common {
namespace cuda {

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__device__ void map_node_ids(const IdType *const global,
                             IdType *const new_global, const size_t num_input,
                             const DeviceOrderedHashTable &table) {
  assert(BLOCK_SIZE == blockDim.x);

  using Bucket = typename OrderedHashTable::BucketO2N;

  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = min(TILE_SIZE * (blockIdx.x + 1), num_input);

  for (size_t idx = threadIdx.x + block_start; idx < block_end;
       idx += BLOCK_SIZE) {
    const Bucket &bucket = *table.SearchO2N(global[idx]);
    new_global[idx] = bucket.local;
  }
}

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void map_edge_ids(const IdType *const global_src,
                             IdType *const new_global_src,
                             const IdType *const global_dst,
                             IdType *const new_global_dst,
                             const size_t num_edges,
                             DeviceOrderedHashTable table) {
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

void GPUMapEdges(const IdType *const global_src, IdType *const new_global_src,
                 const IdType *const global_dst, IdType *const new_global_dst,
                 const size_t num_edges, DeviceOrderedHashTable table,
                 Context ctx, StreamHandle stream) {
  const size_t num_tiles = RoundUpDiv(num_edges, Constant::kCudaTileSize);
  const dim3 grid(num_tiles, 2);
  const dim3 block(Constant::kCudaBlockSize);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  map_edge_ids<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(global_src, new_global_src, global_dst,
                                      new_global_dst, num_edges, table);
  Device::Get(ctx)->StreamSync(ctx, stream);
}

}  // namespace cuda
}  // namespace common
}  // namespace samgraph
