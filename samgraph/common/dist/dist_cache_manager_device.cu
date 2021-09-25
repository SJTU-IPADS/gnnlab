#include <cuda_runtime.h>

#include <cub/cub.cuh>

#include "../common.h"
#include "../constant.h"
#include "../device.h"
#include "../function.h"
#include "../logging.h"
#include "../run_config.h"
#include "../timer.h"
#include "dist_cache_manager.h"

namespace samgraph {
namespace common {
namespace dist {

namespace {

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

}  // namespace

void DistCacheManager::CombineMissData(void *output, const void *miss,
                                      const IdType *miss_dst_index,
                                      const size_t num_miss,
                                      StreamHandle stream) {
  LOG(DEBUG) << "DistCacheManager::CombineMissData():  num_miss " << num_miss;
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

void DistCacheManager::CombineCacheData(void *output,
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

}  // namespace dist
}  // namespace common
}  // namespace samgraph
