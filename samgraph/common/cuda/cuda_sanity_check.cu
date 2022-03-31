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

#include "../common.h"
#include "../constant.h"
#include "../device.h"
#include "cuda_function.h"

namespace samgraph {
namespace common {
namespace cuda {

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void batch_sanity_check(IdType *map, const IdType *input,
                                   const size_t num_input) {
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_input) {
      if (map[input[index]] > 0) {
        printf("duplicate batch input");
      }
      assert(map[input[index]] == 0);
      map[input[index]] = 1;
    }
  }
}

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void list_sanity_check(const IdType *input, const size_t num_input,
                                  IdType invalid_val) {
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_input) {
      assert(input[index] != invalid_val);
    }
  }
}

void GPUBatchSanityCheck(IdType *map, const IdType *input,
                         const size_t num_input, Context ctx,
                         StreamHandle stream) {
  auto device = Device::Get(ctx);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  const uint32_t num_tiles = RoundUpDiv(num_input, Constant::kCudaTileSize);
  const dim3 grid(num_tiles);
  const dim3 block(Constant::kCudaBlockSize);

  batch_sanity_check<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(map, input, num_input);

  device->StreamSync(ctx, stream);
}

void GPUSanityCheckList(const IdType *input, size_t num_input,
                        IdType invalid_val, Context ctx, StreamHandle stream) {
  auto device = Device::Get(ctx);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  const uint32_t num_tiles = RoundUpDiv(num_input, Constant::kCudaTileSize);
  const dim3 grid(num_tiles);
  const dim3 block(Constant::kCudaBlockSize);

  list_sanity_check<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(input, num_input, invalid_val);

  device->StreamSync(ctx, stream);
}

}  // namespace cuda
}  // namespace common
}  // namespace samgraph
