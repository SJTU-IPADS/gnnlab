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
    size_t *block_iter) {
  assert(BLOCK_SIZE == blockDim.x);

  using BlockReduce = typename cub::BlockReduce<size_t, BLOCK_SIZE>;

  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  size_t count = 0;
  IdType thread_iter = TILE_SIZE / BLOCK_SIZE;
#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_input) {
      count += indptr[inputs[index]+1] - indptr[inputs[index]];
    }
  }
  thread_iter += count + 1;

  __shared__ typename BlockReduce::TempStorage temp_space1;
  __shared__ typename BlockReduce::TempStorage temp_space2;

  count = BlockReduce(temp_space1).Sum(count);
  thread_iter = BlockReduce(temp_space2).Reduce(thread_iter, cub::Max());

  if (threadIdx.x == 0) {
    item_prefix[blockIdx.x] = count;
    block_iter[blockIdx.x] = thread_iter;
    if (blockIdx.x == 0) {
      item_prefix[gridDim.x] = 0;
    }
  }
}

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void compact_edge(
    const IdType *const indptr, const IdType *const indices,
    const IdType *const inputs, const size_t num_input,
    const size_t *const item_prefix, const size_t *const block_iter, 
    IdType* output) {
  assert(BLOCK_SIZE == blockDim.x);

  using FlagType = IdType;
  using BlockScan = typename cub::BlockScan<FlagType, BLOCK_SIZE>;

  constexpr const size_t VALS_PER_THREAD = TILE_SIZE / BLOCK_SIZE;

  __shared__ typename BlockScan::TempStorage temp_space;

  const size_t offset = item_prefix[blockIdx.x];

  BlockPrefixCallbackOp<FlagType> prefix_op(0);

  IdType cur_index = threadIdx.x + TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);
  IdType cur_j = 0;
  IdType cur_off = 0;
  IdType cur_len = 0;
  const size_t thread_end = (num_input < block_end) ? num_input : block_end;
  if (cur_index < thread_end) {
    auto orig_id = inputs[cur_index];
    cur_off = indptr[orig_id];
    cur_len = indptr[orig_id + 1] - cur_off;
  }
  for (IdType iter = 0; iter < block_iter[blockIdx.x]; iter++) {
    FlagType flag = 0;
    IdType nbr_orig_id = Constant::kEmptyKey;
    if (cur_index >= thread_end) {
    } else if (cur_j >= cur_len) {
      cur_index += BLOCK_SIZE;
      if (cur_index < thread_end) {
        auto orig_id = inputs[cur_index];
        cur_off = indptr[orig_id];
        cur_len = indptr[orig_id + 1] - cur_off;
        cur_j = 0;
      }
    } else {
      nbr_orig_id = indices[cur_j + cur_off];
      flag = 1;
      cur_j++;
    }

    BlockScan(temp_space).ExclusiveSum(flag, flag, prefix_op);
    __syncthreads();
    
    if (nbr_orig_id != Constant::kEmptyKey) {
      const IdType pos = offset + flag;
      output[pos] = nbr_orig_id;
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
  size_t *item_prefix = static_cast<size_t *>(
      sampler_device->AllocWorkspace(ctx, 2 * sizeof(size_t) * (grid.x + 1)));
  size_t *block_iter = static_cast<size_t *>(item_prefix + grid.x + 1);
  LOG(DEBUG) << "GPUExtractNeighbour: cuda item_prefix malloc "
             << ToReadableSize(sizeof(size_t) * (grid.x + 1));

  count_edge<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(
          indptr, input, num_input, item_prefix, block_iter);
  sampler_device->StreamSync(ctx, stream);
  double count_edge_time = t1.Passed();

  Timer t2;
  size_t workspace_bytes;
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      nullptr, workspace_bytes, static_cast<size_t *>(nullptr),
      static_cast<size_t *>(nullptr), grid.x + 1, cu_stream));
  sampler_device->StreamSync(ctx, stream);

  void *workspace = sampler_device->AllocWorkspace(ctx, workspace_bytes);
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(workspace, workspace_bytes,
                                          item_prefix, item_prefix, grid.x + 1,
                                          cu_stream));
  sampler_device->StreamSync(ctx, stream);
  LOG(DEBUG) << "GPUExtractNeighbour: cuda workspace malloc "
             << ToReadableSize(workspace_bytes);

  sampler_device->CopyDataFromTo(item_prefix + grid.x, 0, num_out, 0, sizeof(size_t), ctx, CPU(), stream);
  sampler_device->StreamSync(ctx, stream);

  output = static_cast<IdType *>(sampler_device->AllocWorkspace(ctx, sizeof(IdType) * *num_out));
  LOG(DEBUG) << "GPUExtractNeighbour: num output: "
             << num_out;

  compact_edge<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(
          indptr, indices, input, num_input, item_prefix, block_iter, output);
  sampler_device->StreamSync(ctx, stream);
  double compact_edge_time = t2.Passed();

  sampler_device->FreeWorkspace(ctx, workspace);
  sampler_device->FreeWorkspace(ctx, item_prefix);

  LOG(DEBUG) << "GPUExtractNeighbour: succeed ";
}

}  // namespace cuda
}  // namespace common
}  // namespace samgraph