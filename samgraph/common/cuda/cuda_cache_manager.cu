#include "cuda_cache_manager.h"

#include <cuda_runtime.h>

#include "../common.h"
#include "../constant.h"
#include "../device.h"
#include "../function.h"
#include "../logging.h"
#include "../run_config.h"
#include "../timer.h"

namespace samgraph {
namespace common {
namespace cuda {

namespace {

template <typename T>
void extract_miss_data(const IdType *hashtable, void *output_miss,
                       IdType *output_miss_index, size_t *num_output_miss,
                       IdType *output_cache_src_index,
                       IdType *output_cache_dst_index, size_t *num_output_cache,
                       const IdType *index, const size_t num_index,
                       const void *src, size_t dim) {
  T *output_miss_data = reinterpret_cast<T *>(output_miss);
  const T *src_data = reinterpret_cast<const T *>(src);

  // Calculate the number of missed data
  size_t tmp_num_output_miss = 0;
  size_t next_output_miss_index = 0;
  size_t next_output_cache_index = 0;

#pragma omp parallel for num_threads(RunConfig::kOMPThreadNum) reduction(+:tmp_num_output_miss)
  for (size_t i = 0; i < num_index; i++) {
    size_t input_index = index[i];
    if (hashtable[input_index] == Constant::kEmptyKey) {
      tmp_num_output_miss++;
      size_t output_index;
#pragma omp critical
      { output_index = next_output_miss_index++; }

      output_miss_index[output_index] = i;
#pragma omp simd
      for (size_t j = 0; j < dim; j++) {
        output_miss_data[output_index * dim + j] =
            src_data[input_index * dim + j];
      }
    } else {
      size_t output_index;

#pragma omp critical
      { output_index = next_output_cache_index++; }

      output_cache_src_index[output_index] = hashtable[input_index];
      output_cache_dst_index[output_index] = i;
    }
  }

  CHECK_EQ(next_output_miss_index, tmp_num_output_miss);
  CHECK_EQ(num_index, tmp_num_output_miss + next_output_cache_index);
  *num_output_miss = tmp_num_output_miss;
  *num_output_cache = next_output_cache_index;

  LOG(DEBUG) << "extract_miss_data "
             << "num_output_miss " << tmp_num_output_miss
             << " num_output_cache " << next_output_cache_index;
}

template <typename T>
__global__ void combine_miss_data(void *output, const void *miss,
                                  const IdType *miss_index,
                                  const size_t num_miss, const size_t dim) {
  T *output_data = reinterpret_cast<T *>(output);
  const T *miss_data = reinterpret_cast<const T *>(miss);

  size_t src_idx = blockIdx.x * blockDim.y + threadIdx.y;
  const size_t stride = blockDim.y * gridDim.x;

  while (src_idx < num_miss) {
    size_t col = threadIdx.x;
    const size_t dst_idx = miss_index[src_idx];
    while (col < dim) {
      output_data[dst_idx * dim + col] = miss_data[src_idx * dim + col];
      col += blockDim.x;
    }

    src_idx += stride;
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

} // namespace

GPUCacheManager::GPUCacheManager(Context ctx, const void *all_data,
                                 DataType dtype, size_t dim,
                                 const IdType *nodes, size_t num_nodes,
                                 double cache_percentage)
    : _ctx(ctx), _cache_percentage(cache_percentage), _num_nodes(num_nodes),
      _num_cached_nodes(num_nodes * cache_percentage), _dtype(dtype), _dim(dim),
      _all_data(all_data) {
  Timer t;

  _cache_nbytes = GetTensorBytes(_dtype, {_num_cached_nodes, _dim});

  auto cpu_device = Device::Get(CPU());
  auto gpu_device = Device::Get(_ctx);
  _cpu_hashtable = static_cast<IdType *>(
      cpu_device->AllocDataSpace(CPU(), sizeof(IdType) * _num_nodes));

  void *tmp_cpu_data = cpu_device->AllocDataSpace(CPU(), _cache_nbytes);
  _cache_gpu_data = gpu_device->AllocDataSpace(_ctx, _cache_nbytes);

  // 1. Initialize the cpu hashtable
#pragma omp parallel for num_threads(RunConfig::kOMPThreadNum)
  for (size_t i = 0; i < _num_nodes; i++) {
    _cpu_hashtable[i] = Constant::kEmptyKey;
  }

  // 2. Populate the cpu hashtable
#pragma omp parallel for num_threads(RunConfig::kOMPThreadNum)
  for (size_t i = 0; i < _num_cached_nodes; i++) {
    _cpu_hashtable[nodes[i]] = i;
  }

  // 3. Populate the cache in cpu memory
  cpu::CPUExtract(tmp_cpu_data, _all_data, nodes, _num_cached_nodes, _dim,
                  _dtype);

  // 4. Copy the cache from the cpu memory to gpu memory
  gpu_device->CopyDataFromTo(tmp_cpu_data, 0, _cache_gpu_data, 0, _cache_nbytes,
                             CPU(), _ctx);

  // 5. Free the cpu tmp cache data
  cpu_device->FreeDataSpace(CPU(), tmp_cpu_data);

  LOG(INFO) << "GPU cache: " << _num_cached_nodes << " / " << _num_nodes
            << " nodes ( " << ToPercentage(_cache_percentage) << " | "
            << ToReadableSize(_cache_nbytes) << " | " << t.Passed()
            << " secs )";
}

GPUCacheManager::~GPUCacheManager() {
  auto cpu_device = Device::Get(CPU());
  auto gpu_device = Device::Get(_ctx);

  cpu_device->FreeDataSpace(CPU(), _cpu_hashtable);
  gpu_device->FreeDataSpace(_ctx, _cache_gpu_data);
}

void GPUCacheManager::ExtractMissData(
    void *output_miss, IdType *output_miss_index, size_t *num_output_miss,
    IdType *output_cache_src_index, IdType *output_cache_dst_index,
    size_t *num_output_cache, const IdType *index, const size_t num_index) {

  switch (_dtype) {
  case kF32:
    extract_miss_data<float>(_cpu_hashtable, output_miss, output_miss_index,
                             num_output_miss, output_cache_src_index,
                             output_cache_dst_index, num_output_cache, index,
                             num_index, _all_data, _dim);
    break;
  case kF64:
    extract_miss_data<double>(_cpu_hashtable, output_miss, output_miss_index,
                              num_output_miss, output_cache_src_index,
                              output_cache_dst_index, num_output_cache, index,
                              num_index, _all_data, _dim);
    break;
  case kF16:
    extract_miss_data<short>(_cpu_hashtable, output_miss, output_miss_index,
                             num_output_miss, output_cache_src_index,
                             output_cache_dst_index, num_output_cache, index,
                             num_index, _all_data, _dim);
    break;
  case kU8:
    extract_miss_data<uint8_t>(_cpu_hashtable, output_miss, output_miss_index,
                               num_output_miss, output_cache_src_index,
                               output_cache_dst_index, num_output_cache, index,
                               num_index, _all_data, _dim);
    break;
  case kI32:
    extract_miss_data<int32_t>(_cpu_hashtable, output_miss, output_miss_index,
                               num_output_miss, output_cache_src_index,
                               output_cache_dst_index, num_output_cache, index,
                               num_index, _all_data, _dim);
    break;
  case kI64:
    extract_miss_data<int64_t>(_cpu_hashtable, output_miss, output_miss_index,
                               num_output_miss, output_cache_src_index,
                               output_cache_dst_index, num_output_cache, index,
                               num_index, _all_data, _dim);
    break;
  default:
    CHECK(0);
  }
}

void GPUCacheManager::CombineMissData(void *output, const void *miss,
                                      const IdType *miss_index,
                                      const size_t num_miss,
                                      StreamHandle stream) {

  LOG(DEBUG) << "GPUCacheManager::CombineMissData():  num_miss " << num_miss;

  auto device = Device::Get(_ctx);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  dim3 block(256, 1);
  while (static_cast<size_t>(block.x) >= 2 * _dim) {
    block.x /= 2;
    block.y *= 2;
  }
  const dim3 grid((num_miss + block.y - 1) / block.y);

  switch (_dtype) {
  case kF32:
    combine_miss_data<float><<<grid, block, 0, cu_stream>>>(
        output, miss, miss_index, num_miss, _dim);
    break;
  case kF64:
    combine_miss_data<double><<<grid, block, 0, cu_stream>>>(
        output, miss, miss_index, num_miss, _dim);
    break;
  case kF16:
    combine_miss_data<short><<<grid, block, 0, cu_stream>>>(
        output, miss, miss_index, num_miss, _dim);
    break;
  case kU8:
    combine_miss_data<uint8_t><<<grid, block, 0, cu_stream>>>(
        output, miss, miss_index, num_miss, _dim);
    break;
  case kI32:
    combine_miss_data<int32_t><<<grid, block, 0, cu_stream>>>(
        output, miss, miss_index, num_miss, _dim);
    break;
  case kI64:
    combine_miss_data<int64_t><<<grid, block, 0, cu_stream>>>(
        output, miss, miss_index, num_miss, _dim);
    break;
  default:
    CHECK(0);
  }

  device->StreamSync(_ctx, stream);
}

void GPUCacheManager::CombineCacheData(void *output,
                                       const IdType *cache_src_index,
                                       const IdType *cache_dst_index,
                                       const size_t num_cache,
                                       StreamHandle stream) {
  CHECK_LE(num_cache, _num_cached_nodes);

  auto device = Device::Get(_ctx);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  dim3 block(256, 1);
  while (static_cast<size_t>(block.x) >= 2 * _dim) {
    block.x /= 2;
    block.y *= 2;
  }
  const dim3 grid((num_cache + block.y - 1) / block.y);

  switch (_dtype) {
  case kF32:
    combine_cache_data<float><<<grid, block, 0, cu_stream>>>(
        output, cache_src_index, cache_dst_index, num_cache, _cache_gpu_data,
        _dim);
    break;
  case kF64:
    combine_cache_data<double><<<grid, block, 0, cu_stream>>>(
        output, cache_src_index, cache_dst_index, num_cache, _cache_gpu_data,
        _dim);
    break;
  case kF16:
    combine_cache_data<short><<<grid, block, 0, cu_stream>>>(
        output, cache_src_index, cache_dst_index, num_cache, _cache_gpu_data,
        _dim);
    break;
  case kU8:
    combine_cache_data<uint8_t><<<grid, block, 0, cu_stream>>>(
        output, cache_src_index, cache_dst_index, num_cache, _cache_gpu_data,
        _dim);
    break;
  case kI32:
    combine_cache_data<int32_t><<<grid, block, 0, cu_stream>>>(
        output, cache_src_index, cache_dst_index, num_cache, _cache_gpu_data,
        _dim);
    break;
  case kI64:
    combine_cache_data<int64_t><<<grid, block, 0, cu_stream>>>(
        output, cache_src_index, cache_dst_index, num_cache, _cache_gpu_data,
        _dim);
    break;
  default:
    CHECK(0);
  }

  device->StreamSync(_ctx, stream);
}

} // namespace cuda
} // namespace common
} // namespace samgraph