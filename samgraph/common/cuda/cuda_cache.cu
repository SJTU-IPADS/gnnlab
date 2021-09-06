#include <cub/cub.cuh>

#include "cuda_function.h"
#include "cuda_utils.h"
#include "../device.h"
#include "../common.h"
#include "../constant.h"

namespace samgraph {
namespace common {
namespace cuda {

namespace {

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void count_miss_cache(const IdType *hashtable, const IdType *nodes,
                                 const size_t num_nodes, IdType *miss_counts,
                                 IdType *cache_counts) {
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  using BlockReduce = typename cub::BlockReduce<IdType, BLOCK_SIZE>;

  IdType miss_count = 0;
  IdType cache_count = 0;

#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_nodes) {
      if (hashtable[nodes[index]] == Constant::kEmptyKey) {
        miss_count++;
      } else {
        cache_count++;
      }
    }
  }

  __shared__ typename BlockReduce::TempStorage temp_miss_space;
  __shared__ typename BlockReduce::TempStorage temp_cache_space;

  miss_count = BlockReduce(temp_miss_space).Sum(miss_count);
  cache_count = BlockReduce(temp_cache_space).Sum(cache_count);

  if (threadIdx.x == 0) {
    miss_counts[blockIdx.x] = miss_count;
    cache_counts[blockIdx.x] = cache_count;
    if (blockIdx.x == 0) {
      miss_counts[gridDim.x] = 0;
      cache_counts[gridDim.x] = 0;
    }
  }
}

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void get_miss_index(const IdType *hashtable, const IdType *nodes,
                               const size_t num_nodes,
                               IdType *output_miss_dst_index,
                               IdType *output_miss_src_index,
                               const IdType *miss_counts_prefix) {
  using FlagType = IdType;
  using BlockScan = typename cub::BlockScan<FlagType, BLOCK_SIZE>;

  constexpr const IdType VALS_PER_THREAD = TILE_SIZE / BLOCK_SIZE;
  __shared__ typename BlockScan::TempStorage temp_space;
  BlockPrefixCallbackOp<FlagType> prefix_op(0);

  const IdType offset = miss_counts_prefix[blockIdx.x];

  for (IdType i = 0; i < VALS_PER_THREAD; ++i) {
    const IdType index = threadIdx.x + i * BLOCK_SIZE + blockIdx.x * TILE_SIZE;

    FlagType flag;
    if (index < num_nodes && hashtable[nodes[index]] == Constant::kEmptyKey) {
      flag = 1;
    } else {
      flag = 0;
    }

    BlockScan(temp_space).ExclusiveSum(flag, flag, prefix_op);
    __syncthreads();

    if (index < num_nodes && hashtable[nodes[index]] == Constant::kEmptyKey) {
      const IdType pos = offset + flag;
      assert(pos < num_nodes);
      // new node ID in subgraph
      output_miss_dst_index[pos] = index;
      // old node ID in original graph
      output_miss_src_index[pos] = nodes[index];
    }
  }

  // if (threadIdx.x == 0 && blockIdx.x == 0) {
  //   printf("miss count %u, %u\n", miss_counts_prefix[gridDim.x],
  //          miss_counts_prefix[gridDim.x - 1]);
  // }
}

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void get_cache_index(const IdType *hashtable, const IdType *nodes,
                                const size_t num_nodes,
                                IdType *output_cache_dst_index,
                                IdType *output_cache_src_index,
                                const IdType *cache_counts_prefix) {
  using FlagType = IdType;
  using BlockScan = typename cub::BlockScan<FlagType, BLOCK_SIZE>;

  constexpr const IdType VALS_PER_THREAD = TILE_SIZE / BLOCK_SIZE;
  __shared__ typename BlockScan::TempStorage temp_space;
  BlockPrefixCallbackOp<FlagType> prefix_op(0);

  const IdType offset = cache_counts_prefix[blockIdx.x];

  for (IdType i = 0; i < VALS_PER_THREAD; ++i) {
    const IdType index = threadIdx.x + i * BLOCK_SIZE + blockIdx.x * TILE_SIZE;

    FlagType flag;
    if (index < num_nodes && hashtable[nodes[index]] != Constant::kEmptyKey) {
      flag = 1;
    } else {
      flag = 0;
    }

    BlockScan(temp_space).ExclusiveSum(flag, flag, prefix_op);
    __syncthreads();

    if (index < num_nodes && hashtable[nodes[index]] != Constant::kEmptyKey) {
      const IdType pos = offset + flag;
      // new node ID in subgraph
      output_cache_dst_index[pos] = index;
      // old node ID in original graph
      output_cache_src_index[pos] = hashtable[nodes[index]];
    }
  }

  // if (threadIdx.x == 0 && blockIdx.x == 0) {
  //   printf("cache count %u, %u\n", cache_counts_prefix[gridDim.x],
  //          cache_counts_prefix[gridDim.x - 1]);
  // }
}

} // namespace

void GetMissCacheIndex(
    IdType *sampler_gpu_hashtable, Context sampler_ctx,
    IdType *output_miss_src_index, IdType *output_miss_dst_index,
    size_t *num_output_miss, IdType *output_cache_src_index,
    IdType *output_cache_dst_index, size_t *num_output_cache,
    const IdType *nodes, const size_t num_nodes, StreamHandle stream) {
  const size_t num_tiles = RoundUpDiv(num_nodes, Constant::kCudaTileSize);
  const dim3 grid(num_tiles);
  const dim3 block(Constant::kCudaBlockSize);

  auto sampler_device = Device::Get(sampler_ctx);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  sampler_device->SetDevice(sampler_ctx);

  IdType *miss_prefix_counts =
      static_cast<IdType *>(sampler_device->AllocWorkspace(
          sampler_ctx, sizeof(IdType) * (grid.x + 1)));
  IdType *cache_prefix_counts =
      static_cast<IdType *>(sampler_device->AllocWorkspace(
          sampler_ctx, sizeof(IdType) * (grid.x + 1)));

  // LOG(DEBUG) << "GetMissCacheIndex num nodes " << num_nodes;

  CUDA_CALL(cudaSetDevice(sampler_ctx.device_id));
  count_miss_cache<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(sampler_gpu_hashtable, nodes, num_nodes,
                                      miss_prefix_counts, cache_prefix_counts);
  sampler_device->StreamSync(sampler_ctx, stream);

  size_t workspace_bytes;
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      nullptr, workspace_bytes, static_cast<IdType *>(nullptr),
      static_cast<IdType *>(nullptr), grid.x + 1, cu_stream));
  sampler_device->StreamSync(sampler_ctx, stream);

  void *workspace =
      sampler_device->AllocWorkspace(sampler_ctx, workspace_bytes);
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      workspace, workspace_bytes, miss_prefix_counts, miss_prefix_counts,
      grid.x + 1, cu_stream));
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      workspace, workspace_bytes, cache_prefix_counts, cache_prefix_counts,
      grid.x + 1, cu_stream));
  sampler_device->StreamSync(sampler_ctx, stream);

  get_miss_index<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(
          sampler_gpu_hashtable, nodes, num_nodes, output_miss_dst_index,
          output_miss_src_index, miss_prefix_counts);
  sampler_device->StreamSync(sampler_ctx, stream);

  get_cache_index<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(
          sampler_gpu_hashtable, nodes, num_nodes, output_cache_dst_index,
          output_cache_src_index, cache_prefix_counts);
  sampler_device->StreamSync(sampler_ctx, stream);

  IdType num_miss;
  IdType num_cache;
  sampler_device->CopyDataFromTo(miss_prefix_counts + grid.x, 0, &num_miss, 0,
                                 sizeof(IdType), sampler_ctx, CPU(), stream);
  sampler_device->CopyDataFromTo(cache_prefix_counts + grid.x, 0, &num_cache, 0,
                                 sizeof(IdType), sampler_ctx, CPU(), stream);
  sampler_device->StreamSync(sampler_ctx, stream);

  *num_output_miss = num_miss;
  *num_output_cache = num_cache;

  sampler_device->FreeWorkspace(sampler_ctx, workspace);
  sampler_device->FreeWorkspace(sampler_ctx, cache_prefix_counts);
  sampler_device->FreeWorkspace(sampler_ctx, miss_prefix_counts);
}


}  // namespace cuda
}  // namespace common
}  // namespace samgraph
