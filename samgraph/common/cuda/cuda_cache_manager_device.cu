#include <cuda_runtime.h>

#include <cub/cub.cuh>

#include "../common.h"
#include "../constant.h"
#include "../device.h"
#include "../function.h"
#include "../logging.h"
#include "../run_config.h"
#include "../timer.h"
#include "cuda_cache_manager.h"
#include "cuda_utils.h"

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

template <typename T>
__global__ void combine_miss_data(void *output, const void *miss,
                                  const IdType *miss_dst_index,
                                  const size_t num_miss, const size_t dim) {
  T *output_data = reinterpret_cast<T *>(output);
  const T *miss_data = reinterpret_cast<const T *>(miss);

  size_t i = blockIdx.x * blockDim.y + threadIdx.y;
  const size_t stride = blockDim.y * gridDim.x;

  /** SXN: why need a loop?*/
  /** SXN: ans: this loop is not necessary*/
  while (i < num_miss) {
    size_t col = threadIdx.x;
    const size_t dst_idx = miss_dst_index[i];
    while (col < dim) {
      output_data[dst_idx * dim + col] = miss_data[i * dim + col];
      col += blockDim.x;
    }

    i += stride;
  }
}

template <typename T>
__global__ void combine_cache_data(void *output, const IdType *cache_src_index,
                                   const IdType *cache_dst_index,
                                   const size_t num_cache, const void *cache,
                                   size_t dim) {
  T *output_data = reinterpret_cast<T *>(output);
  const T *cache_data = reinterpret_cast<const T *>(cache);

  size_t i = blockIdx.x * blockDim.y + threadIdx.y;
  const size_t stride = blockDim.y * gridDim.x;

  while (i < num_cache) {
    size_t col = threadIdx.x;
    const size_t src_idx = cache_src_index[i];
    const size_t dst_idx = cache_dst_index[i];
    while (col < dim) {
      output_data[dst_idx * dim + col] = cache_data[src_idx * dim + col];
      col += blockDim.x;
    }
    i += stride;
  }
}

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void hashtable_insert_nodes(const IdType *const cached_nodes,
                                       const size_t num_cached_nodes,
                                       IdType* gpu_hashtable) {
  assert(BLOCK_SIZE == blockDim.x);

  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_cached_nodes) {
      gpu_hashtable[cached_nodes[index]] = index;
    }
  }
}
template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void hashtable_reset_nodes(const IdType *const cached_nodes,
                                       const size_t num_cached_nodes,
                                       IdType* gpu_hashtable) {
  assert(BLOCK_SIZE == blockDim.x);

  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_cached_nodes) {
      gpu_hashtable[cached_nodes[index]] = Constant::kEmptyKey;
    }
  }
}

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void hashtable_reset_nodes(const size_t num_total_nodes,
                                      IdType* gpu_hashtable) {
  assert(BLOCK_SIZE == blockDim.x);

  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_total_nodes) {
      gpu_hashtable[index] = Constant::kEmptyKey;
    }
  }
}

}  // namespace

void GPUCacheManager::GetMissCacheIndex(
    IdType *output_miss_src_index, IdType *output_miss_dst_index,
    size_t *num_output_miss, IdType *output_cache_src_index,
    IdType *output_cache_dst_index, size_t *num_output_cache,
    const IdType *nodes, const size_t num_nodes, StreamHandle stream) {
  const size_t num_tiles = RoundUpDiv(num_nodes, Constant::kCudaTileSize);
  const dim3 grid(num_tiles);
  const dim3 block(Constant::kCudaBlockSize);

  auto sampler_device = Device::Get(_sampler_ctx);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  sampler_device->SetDevice(_sampler_ctx);

  IdType *miss_prefix_counts =
      static_cast<IdType *>(sampler_device->AllocWorkspace(
          _sampler_ctx, sizeof(IdType) * (grid.x + 1)));
  IdType *cache_prefix_counts =
      static_cast<IdType *>(sampler_device->AllocWorkspace(
          _sampler_ctx, sizeof(IdType) * (grid.x + 1)));

  // LOG(DEBUG) << "GetMissCacheIndex num nodes " << num_nodes;

  CUDA_CALL(cudaSetDevice(_sampler_ctx.device_id));
  count_miss_cache<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(_sampler_gpu_hashtable, nodes, num_nodes,
                                      miss_prefix_counts, cache_prefix_counts);
  sampler_device->StreamSync(_sampler_ctx, stream);

  size_t workspace_bytes;
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      nullptr, workspace_bytes, static_cast<IdType *>(nullptr),
      static_cast<IdType *>(nullptr), grid.x + 1, cu_stream));
  sampler_device->StreamSync(_sampler_ctx, stream);

  void *workspace =
      sampler_device->AllocWorkspace(_sampler_ctx, workspace_bytes);
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      workspace, workspace_bytes, miss_prefix_counts, miss_prefix_counts,
      grid.x + 1, cu_stream));
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      workspace, workspace_bytes, cache_prefix_counts, cache_prefix_counts,
      grid.x + 1, cu_stream));
  sampler_device->StreamSync(_sampler_ctx, stream);

  get_miss_index<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(
          _sampler_gpu_hashtable, nodes, num_nodes, output_miss_dst_index,
          output_miss_src_index, miss_prefix_counts);
  sampler_device->StreamSync(_sampler_ctx, stream);

  get_cache_index<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(
          _sampler_gpu_hashtable, nodes, num_nodes, output_cache_dst_index,
          output_cache_src_index, cache_prefix_counts);
  sampler_device->StreamSync(_sampler_ctx, stream);

  IdType num_miss;
  IdType num_cache;
  sampler_device->CopyDataFromTo(miss_prefix_counts + grid.x, 0, &num_miss, 0,
                                 sizeof(IdType), _sampler_ctx, CPU(), stream);
  sampler_device->CopyDataFromTo(cache_prefix_counts + grid.x, 0, &num_cache, 0,
                                 sizeof(IdType), _sampler_ctx, CPU(), stream);
  sampler_device->StreamSync(_sampler_ctx, stream);

  *num_output_miss = num_miss;
  *num_output_cache = num_cache;

  sampler_device->FreeWorkspace(_sampler_ctx, workspace);
  sampler_device->FreeWorkspace(_sampler_ctx, cache_prefix_counts);
  sampler_device->FreeWorkspace(_sampler_ctx, miss_prefix_counts);
}

void GPUCacheManager::CombineMissData(void *output, const void *miss,
                                      const IdType *miss_dst_index,
                                      const size_t num_miss,
                                      StreamHandle stream) {
  LOG(DEBUG) << "GPUCacheManager::CombineMissData():  num_miss " << num_miss;
  if (num_miss == 0) return;

  auto device = Device::Get(_trainer_ctx);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  dim3 block(256, 1);
  while (static_cast<size_t>(block.x) >= 2 * _dim) {
    block.x /= 2;
    block.y *= 2;
  }
  const dim3 grid(RoundUpDiv(num_miss, static_cast<size_t>(block.y)));

  switch (_dtype) {
    case kF32:
      combine_miss_data<float><<<grid, block, 0, cu_stream>>>(
          output, miss, miss_dst_index, num_miss, _dim);
      break;
    case kF64:
      combine_miss_data<double><<<grid, block, 0, cu_stream>>>(
          output, miss, miss_dst_index, num_miss, _dim);
      break;
    case kF16:
      combine_miss_data<short><<<grid, block, 0, cu_stream>>>(
          output, miss, miss_dst_index, num_miss, _dim);
      break;
    case kU8:
      combine_miss_data<uint8_t><<<grid, block, 0, cu_stream>>>(
          output, miss, miss_dst_index, num_miss, _dim);
      break;
    case kI32:
      combine_miss_data<int32_t><<<grid, block, 0, cu_stream>>>(
          output, miss, miss_dst_index, num_miss, _dim);
      break;
    case kI64:
      combine_miss_data<int64_t><<<grid, block, 0, cu_stream>>>(
          output, miss, miss_dst_index, num_miss, _dim);
      break;
    default:
      CHECK(0);
  }

  device->StreamSync(_trainer_ctx, stream);
}

void GPUCacheManager::CombineCacheData(void *output,
                                       const IdType *cache_src_index,
                                       const IdType *cache_dst_index,
                                       const size_t num_cache,
                                       StreamHandle stream) {
  CHECK_LE(num_cache, _num_cached_nodes);

  auto device = Device::Get(_trainer_ctx);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  dim3 block(256, 1);
  while (static_cast<size_t>(block.x) >= 2 * _dim) {
    block.x /= 2;
    block.y *= 2;
  }
  const dim3 grid(RoundUpDiv(num_cache, static_cast<size_t>(block.y)));

  switch (_dtype) {
    case kF32:
      combine_cache_data<float><<<grid, block, 0, cu_stream>>>(
          output, cache_src_index, cache_dst_index, num_cache,
          _trainer_cache_data, _dim);
      break;
    case kF64:
      combine_cache_data<double><<<grid, block, 0, cu_stream>>>(
          output, cache_src_index, cache_dst_index, num_cache,
          _trainer_cache_data, _dim);
      break;
    case kF16:
      combine_cache_data<short><<<grid, block, 0, cu_stream>>>(
          output, cache_src_index, cache_dst_index, num_cache,
          _trainer_cache_data, _dim);
      break;
    case kU8:
      combine_cache_data<uint8_t><<<grid, block, 0, cu_stream>>>(
          output, cache_src_index, cache_dst_index, num_cache,
          _trainer_cache_data, _dim);
      break;
    case kI32:
      combine_cache_data<int32_t><<<grid, block, 0, cu_stream>>>(
          output, cache_src_index, cache_dst_index, num_cache,
          _trainer_cache_data, _dim);
      break;
    case kI64:
      combine_cache_data<int64_t><<<grid, block, 0, cu_stream>>>(
          output, cache_src_index, cache_dst_index, num_cache,
          _trainer_cache_data, _dim);
      break;
    default:
      CHECK(0);
  }

  device->StreamSync(_trainer_ctx, stream);
}

void GPUDynamicCacheManager::GetMissCacheIndex(
    IdType *output_miss_src_index, IdType *output_miss_dst_index,
    size_t *num_output_miss, IdType *output_cache_src_index,
    IdType *output_cache_dst_index, size_t *num_output_cache,
    const IdType *nodes, const size_t num_nodes, StreamHandle stream) {
  const size_t num_tiles = RoundUpDiv(num_nodes, Constant::kCudaTileSize);
  const dim3 grid(num_tiles);
  const dim3 block(Constant::kCudaBlockSize);

  auto sampler_device = Device::Get(_sampler_ctx);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  sampler_device->SetDevice(_sampler_ctx);

  IdType *miss_prefix_counts =
      static_cast<IdType *>(sampler_device->AllocWorkspace(
          _sampler_ctx, sizeof(IdType) * (grid.x + 1)));
  IdType *cache_prefix_counts =
      static_cast<IdType *>(sampler_device->AllocWorkspace(
          _sampler_ctx, sizeof(IdType) * (grid.x + 1)));

  // LOG(DEBUG) << "GetMissCacheIndex num nodes " << num_nodes;

  CUDA_CALL(cudaSetDevice(_sampler_ctx.device_id));
  count_miss_cache<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(_sampler_gpu_hashtable, nodes, num_nodes,
                                      miss_prefix_counts, cache_prefix_counts);
  sampler_device->StreamSync(_sampler_ctx, stream);

  size_t workspace_bytes;
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      nullptr, workspace_bytes, static_cast<IdType *>(nullptr),
      static_cast<IdType *>(nullptr), grid.x + 1, cu_stream));
  sampler_device->StreamSync(_sampler_ctx, stream);

  void *workspace =
      sampler_device->AllocWorkspace(_sampler_ctx, workspace_bytes);
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      workspace, workspace_bytes, miss_prefix_counts, miss_prefix_counts,
      grid.x + 1, cu_stream));
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      workspace, workspace_bytes, cache_prefix_counts, cache_prefix_counts,
      grid.x + 1, cu_stream));
  sampler_device->StreamSync(_sampler_ctx, stream);

  get_miss_index<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(
          _sampler_gpu_hashtable, nodes, num_nodes, output_miss_dst_index,
          output_miss_src_index, miss_prefix_counts);
  sampler_device->StreamSync(_sampler_ctx, stream);

  get_cache_index<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(
          _sampler_gpu_hashtable, nodes, num_nodes, output_cache_dst_index,
          output_cache_src_index, cache_prefix_counts);
  sampler_device->StreamSync(_sampler_ctx, stream);

  IdType num_miss;
  IdType num_cache;
  sampler_device->CopyDataFromTo(miss_prefix_counts + grid.x, 0, &num_miss, 0,
                                 sizeof(IdType), _sampler_ctx, CPU(), stream);
  sampler_device->CopyDataFromTo(cache_prefix_counts + grid.x, 0, &num_cache, 0,
                                 sizeof(IdType), _sampler_ctx, CPU(), stream);
  sampler_device->StreamSync(_sampler_ctx, stream);

  *num_output_miss = num_miss;
  *num_output_cache = num_cache;

  sampler_device->FreeWorkspace(_sampler_ctx, workspace);
  sampler_device->FreeWorkspace(_sampler_ctx, cache_prefix_counts);
  sampler_device->FreeWorkspace(_sampler_ctx, miss_prefix_counts);
}

void GPUDynamicCacheManager::CombineMissData(void *output, const void *miss,
                                             const IdType *miss_dst_index,
                                             const size_t num_miss,
                                             StreamHandle stream) {
  LOG(DEBUG) << "GPUDynamicCacheManager::CombineMissData():  num_miss "
             << num_miss;

  auto device = Device::Get(_trainer_ctx);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  dim3 block(256, 1);
  while (static_cast<size_t>(block.x) >= 2 * _dim) {
    block.x /= 2;
    block.y *= 2;
  }
  const dim3 grid(RoundUpDiv(num_miss, static_cast<size_t>(block.y)));

  switch (_dtype) {
    case kF32:
      combine_miss_data<float><<<grid, block, 0, cu_stream>>>(
          output, miss, miss_dst_index, num_miss, _dim);
      break;
    case kF64:
      combine_miss_data<double><<<grid, block, 0, cu_stream>>>(
          output, miss, miss_dst_index, num_miss, _dim);
      break;
    case kF16:
      combine_miss_data<short><<<grid, block, 0, cu_stream>>>(
          output, miss, miss_dst_index, num_miss, _dim);
      break;
    case kU8:
      combine_miss_data<uint8_t><<<grid, block, 0, cu_stream>>>(
          output, miss, miss_dst_index, num_miss, _dim);
      break;
    case kI32:
      combine_miss_data<int32_t><<<grid, block, 0, cu_stream>>>(
          output, miss, miss_dst_index, num_miss, _dim);
      break;
    case kI64:
      combine_miss_data<int64_t><<<grid, block, 0, cu_stream>>>(
          output, miss, miss_dst_index, num_miss, _dim);
      break;
    default:
      CHECK(0);
  }

  device->StreamSync(_trainer_ctx, stream);
}

void GPUDynamicCacheManager::CombineCacheData(void *output,
                                              const IdType *cache_src_index,
                                              const IdType *cache_dst_index,
                                              const size_t num_cache,
                                              StreamHandle stream) {
  if (_trainer_cache_data == nullptr) {
    CHECK_EQ(num_cache, 0);
    return;
  }
  CHECK_LE(num_cache, _trainer_cache_data->Shape()[0]);
  const void * train_cache_data = _trainer_cache_data->Data();

  auto device = Device::Get(_trainer_ctx);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  dim3 block(256, 1);
  while (static_cast<size_t>(block.x) >= 2 * _dim) {
    block.x /= 2;
    block.y *= 2;
  }
  const dim3 grid(RoundUpDiv(num_cache, static_cast<size_t>(block.y)));

  switch (_dtype) {
    case kF32:
      combine_cache_data<float><<<grid, block, 0, cu_stream>>>(
          output, cache_src_index, cache_dst_index, num_cache,
          train_cache_data, _dim);
      break;
    case kF64:
      combine_cache_data<double><<<grid, block, 0, cu_stream>>>(
          output, cache_src_index, cache_dst_index, num_cache,
          train_cache_data, _dim);
      break;
    case kF16:
      combine_cache_data<short><<<grid, block, 0, cu_stream>>>(
          output, cache_src_index, cache_dst_index, num_cache,
          train_cache_data, _dim);
      break;
    case kU8:
      combine_cache_data<uint8_t><<<grid, block, 0, cu_stream>>>(
          output, cache_src_index, cache_dst_index, num_cache,
          train_cache_data, _dim);
      break;
    case kI32:
      combine_cache_data<int32_t><<<grid, block, 0, cu_stream>>>(
          output, cache_src_index, cache_dst_index, num_cache,
          train_cache_data, _dim);
      break;
    case kI64:
      combine_cache_data<int64_t><<<grid, block, 0, cu_stream>>>(
          output, cache_src_index, cache_dst_index, num_cache,
          train_cache_data, _dim);
      break;
    default:
      CHECK(0);
  }

  device->StreamSync(_trainer_ctx, stream);
}

/**
 * @brief 
 * 
 * @param nodes must be in cpu
 * @param features must be in train gpu
 */
 void GPUDynamicCacheManager::ReplaceCacheGPU(TensorPtr nodes, TensorPtr features, StreamHandle stream) {
  Timer t;

  // _cache_nbytes = GetTensorBytes(_dtype, {_num_cached_nodes, _dim});

  // auto cpu_device = Device::Get(CPU());
  auto sampler_gpu_device = Device::Get(_sampler_ctx);
  // auto trainer_gpu_device = Device::Get(_trainer_ctx);

  // IdType *tmp_cpu_hashtable = static_cast<IdType *>(
  //     cpu_device->AllocDataSpace(CPU(), sizeof(IdType) * _num_nodes));
  // void *tmp_cpu_data = cpu_device->AllocDataSpace(CPU(), _cache_nbytes);

  auto cu_stream = static_cast<cudaStream_t>(stream);

  CHECK_EQ(nodes->Ctx().device_type, _sampler_ctx.device_type);
  CHECK_EQ(nodes->Ctx().device_id, _sampler_ctx.device_id);
  CHECK_EQ(features->Ctx().device_type, _trainer_ctx.device_type);
  CHECK_EQ(features->Ctx().device_id, _trainer_ctx.device_id);

  CHECK_EQ(features->Shape()[0], nodes->Shape()[0]);


  sampler_gpu_device->SetDevice(_sampler_ctx);
  // 1. Initialize the gpu hashtable
  if (_cached_nodes != nullptr) {
    const IdType* old_cached_nodes = static_cast<const IdType*>(_cached_nodes->Data());
    size_t num_old_cached_nodes = _cached_nodes->Shape()[0];
    const size_t old_num_tiles = RoundUpDiv(num_old_cached_nodes, Constant::kCudaTileSize);
    const dim3 old_grid(old_num_tiles);
    const dim3 old_block(Constant::kCudaBlockSize);
    hashtable_reset_nodes<Constant::kCudaBlockSize, Constant::kCudaTileSize>
        <<<old_grid, old_block, 0, cu_stream>>>(
            old_cached_nodes, num_old_cached_nodes, _sampler_gpu_hashtable);
    sampler_gpu_device->StreamSync(_sampler_ctx, stream);
  }
  
  // sampler_gpu_device->CopyDataFromTo(
  //   _cpu_hashtable, 0, _sampler_gpu_hashtable, 0,
  //   sizeof(IdType) * _num_nodes, CPU(), _sampler_ctx, stream);
  // sampler_gpu_device->StreamSync(_sampler_ctx, stream);


  // 2. Populate the cpu hashtable
  const IdType* new_cached_nodes = static_cast<const IdType*>(nodes->Data());
  size_t num_new_cached_nodes = nodes->Shape()[0];
  const size_t num_tiles = RoundUpDiv(num_new_cached_nodes, Constant::kCudaTileSize);
  const dim3 grid(num_tiles);
  const dim3 block(Constant::kCudaBlockSize);
  hashtable_insert_nodes<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(new_cached_nodes, num_new_cached_nodes, _sampler_gpu_hashtable);
  sampler_gpu_device->StreamSync(_sampler_ctx, stream);

  _trainer_cache_data = features;
  _cached_nodes = nodes;

  // 3. Populate the cache in cpu memory
  // cpu::CPUExtract(tmp_cpu_data, _cpu_src_data, nodes, _num_cached_nodes, _dim,
  //                 _dtype);

  // 4. Copy the cache from the cpu memory to gpu memory
  // sampler_gpu_device->CopyDataFromTo(
  //     _cpu_hashtable, 0, _sampler_gpu_hashtable, 0,
  //     sizeof(IdType) * _num_nodes, CPU(), _sampler_ctx);
  // trainer_gpu_device->CopyDataFromTo(tmp_cpu_data, 0, _trainer_cache_data, 0,
  //                                    _cache_nbytes, CPU(), _trainer_ctx);

  // 5. Free the cpu tmp cache data
  // cpu_device->FreeDataSpace(CPU(), tmp_cpu_hashtable);
  // cpu_device->FreeDataSpace(CPU(), tmp_cpu_data);

  LOG(INFO) << "GPU dynamic cache "
            << features->Shape()[0] << " / " << _num_nodes << " nodes ( "
            << ToPercentage(features->Shape()[0] / (float)_num_nodes) << " | "
            << ToReadableSize(features->NumBytes()) << " | " << t.Passed()
            << " secs )";
}

}  // namespace cuda
}  // namespace common
}  // namespace samgraph