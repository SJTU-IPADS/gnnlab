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

#include <curand.h>
#include <curand_kernel.h>

#include <cassert>
#include <chrono>
#include <cstdio>
#include <cub/cub.cuh>

#include "../common.h"
#include "../constant.h"
#include "../device.h"
#include "../logging.h"
#include "../profiler.h"
#include "../timer.h"
#include "cuda_function.h"
#include "cuda_utils.h"

namespace samgraph {
namespace common {
namespace cuda {

namespace {

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void count_edge(
    const IdType *const indptr, const IdType * const inputs,
    const size_t num_input, 
    size_t *item_prefix,
    size_t *) {
  assert(BLOCK_SIZE == blockDim.x);

  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  size_t count = 0;
#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_input) {
      count += indptr[inputs[index]+1] - indptr[inputs[index]];
    }
  }

  item_prefix[blockIdx.x * blockDim.x + threadIdx.x] = count;

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    item_prefix[gridDim.x * blockDim.x] = 0;
  }
}

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void compact_edge(
    const IdType *const indptr, const IdType *const indices,
    const IdType *const inputs, const size_t num_input,
    const size_t *const item_prefix, const size_t *const, 
    IdType* output) {
  assert(BLOCK_SIZE == blockDim.x);

  using FlagType = IdType;

  const size_t thread_offset = item_prefix[blockIdx.x * blockDim.x + threadIdx.x];

  IdType cur_index = threadIdx.x + TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);
  IdType cur_j = 0;
  IdType cur_off = 0;
  IdType cur_len = 0;
  IdType thread_pos = 0;
  const size_t thread_end = (num_input < block_end) ? num_input : block_end;
  if (cur_index < thread_end) {
    auto orig_id = inputs[cur_index];
    cur_off = indptr[orig_id];
    cur_len = indptr[orig_id + 1] - cur_off;
  }
  while (cur_index < thread_end) {
    IdType nbr_orig_id = Constant::kEmptyKey;
    if (cur_j >= cur_len) {
      cur_index += BLOCK_SIZE;
      if (cur_index >= thread_end) break;
      auto orig_id = inputs[cur_index];
      cur_off = indptr[orig_id];
      cur_len = indptr[orig_id + 1] - cur_off;
      cur_j = 0;
    } else {
      nbr_orig_id = indices[cur_j + cur_off];
      output[thread_offset + thread_pos] = nbr_orig_id;
      thread_pos ++;
      cur_j++;
    }
  }
}
}  // namespace

void GPUExtractNeighbour(const IdType *indptr, const IdType *indices,
                    const IdType *input, const size_t num_input,
                    IdType *&output,
                    size_t *num_out, Context ctx, StreamHandle stream,
                    const uint64_t task_key) {
  LOG(DEBUG) << "GPUExtractNeighbour: begin with num_input " << num_input;
  const size_t num_tiles = RoundUpDiv(num_input, Constant::kCudaTileSize);
  const dim3 grid(num_tiles);
  const dim3 block(Constant::kCudaBlockSize);
  
  auto sampler_device = Device::Get(ctx);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  Timer t1;
  size_t n_item_prefix = block.x * grid.x + 1;
  size_t *item_prefix = static_cast<size_t *>(
      sampler_device->AllocWorkspace(ctx, sizeof(size_t) * 2 * n_item_prefix));
  size_t *const item_prefix_out = &item_prefix[n_item_prefix];

  LOG(DEBUG) << "GPUExtractNeighbour: cuda item_prefix malloc "
             << ToReadableSize(sizeof(size_t) * 2 * n_item_prefix);

  count_edge<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(
          indptr, input, num_input, item_prefix, nullptr);
  sampler_device->StreamSync(ctx, stream);
  double count_edge_time = t1.Passed();

  Timer t2;
  size_t workspace_bytes;
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      nullptr, workspace_bytes, static_cast<size_t *>(nullptr),
      static_cast<size_t *>(nullptr), n_item_prefix, cu_stream));
  sampler_device->StreamSync(ctx, stream);

  void *workspace = sampler_device->AllocWorkspace(ctx, workspace_bytes);
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(workspace, workspace_bytes,
                                          item_prefix, item_prefix_out, n_item_prefix,
                                          cu_stream));
  sampler_device->StreamSync(ctx, stream);
  LOG(DEBUG) << "GPUExtractNeighbour: cuda workspace malloc "
             << ToReadableSize(workspace_bytes);

  sampler_device->CopyDataFromTo(&item_prefix_out[n_item_prefix - 1], 0, num_out, 0, sizeof(size_t), ctx, CPU(), stream);
  sampler_device->StreamSync(ctx, stream);

  output = static_cast<IdType *>(sampler_device->AllocWorkspace(ctx, sizeof(IdType) * *num_out));
  LOG(DEBUG) << "GPUExtractNeighbour: num output: "
             << *num_out;

  compact_edge<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(
          indptr, indices, input, num_input, item_prefix, nullptr, output);
  sampler_device->StreamSync(ctx, stream);
  double compact_edge_time = t2.Passed();

  sampler_device->FreeWorkspace(ctx, workspace);
  sampler_device->FreeWorkspace(ctx, item_prefix);

  LOG(DEBUG) << "GPUExtractNeighbour: succeed ";
}

}  // namespace cuda
}  // namespace common
}  // namespace samgraph