#include "cuda_cache.h"

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
void extract_miss_data(const IdType *hashtable, void *output,
                       IdType *output_index, size_t *num_output,
                       const IdType *index, const size_t num_index,
                       const void *src, size_t dim) {
  T *output_data = reinterpret_cast<T *>(output);
  const T *src_data = reinterpret_cast<const T *>(src);

  // Calculate the number of missed data
  size_t tmp_num_output = 0;
  IdType next_output_index = 0;

#pragma omp parallel for num_threads(RunConfig::kOMPThreadNum) reduction(+:num_out)
  for (size_t i = 0; i < num_index; i++) {
    if (hashtable[index[i]] == Constant::kEmptyKey) {
      tmp_num_output++;
      IdType output_index;
#pragma omp critical
      { output_index = next_output_index++; }

#pragma omp simd
      for (size_t j = 0; j < dim; j++) {
        output_data[output_index * dim + j] = src_data[index[i] * dim + j];
      }
    }
  }

  CHECK_EQ(next_output_index, tmp_num_output);
  *num_output = tmp_num_output;
}

template <typename T, size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void combine_cache_data(const IdType *hashtable, char *output,
                                   const char *cache_data, const IdType *index,
                                   const size_t num_index, size_t nbytes) {}

template <typename T, size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void combine_miss_data(char *output, const char *cache_data,
                                  const IdType *index, const size_t num_index,
                                  const IdType *hashtable) {}

} // namespace

GPUCache::GPUCache(Context ctx, const void *all_data, DataType dtype,
                   size_t dim, const IdType *nodes, size_t num_nodes,
                   double cache_percentage)
    : _all_data(all_data) {
  Timer t;

  _ctx = ctx;
  _cache_percentage = cache_percentage;
  _num_nodes = num_nodes;
  _num_cached_nodes = _num_nodes * _cache_percentage;

  _dtype = dtype;
  _dim = dim;

  _cache_nbytes = GetTensorBytes(_dtype, {_num_cached_nodes, _dim});

  auto cpu_device = Device::Get(CPU());
  auto gpu_device = Device::Get(_ctx);
  _cpu_hashtable = static_cast<IdType *>(
      cpu_device->AllocDataSpace(CPU(), sizeof(IdType) * _num_nodes));
  _gpu_hashtable = static_cast<IdType *>(
      gpu_device->AllocDataSpace(_ctx, sizeof(IdType) * _num_nodes));

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
    _cpu_hashtable[nodes[i]] = Constant::kCacheKey;
  }

  // 3. Copy the hashtable from the cpu memory to gpu memory
  gpu_device->CopyDataFromTo(_cpu_hashtable, 0, _gpu_hashtable, 0,
                             sizeof(IdType) * _num_nodes, CPU(), _ctx);

  // 4. Populate the cache in cpu memory
  cpu::CPUExtract(tmp_cpu_data, _all_data, nodes, _num_cached_nodes, _dim,
                  _dtype);

  // 5. Copy the cache from the cpu memory to gpu memory
  gpu_device->CopyDataFromTo(tmp_cpu_data, 0, _cache_gpu_data, 0, _cache_nbytes,
                             CPU(), _ctx);

  // 6. Free the cpu tmp cache data
  cpu_device->FreeDataSpace(CPU(), tmp_cpu_data);

  LOG(INFO) << "GPU cache: " << _num_cached_nodes << " / " << _num_nodes
            << " nodes ( " << ToPercentage(_cache_percentage) << " | "
            << ToReadableSize(_cache_nbytes) << " | " << t.Passed()
            << " secs )";
}

GPUCache::~GPUCache() {
  auto cpu_device = Device::Get(CPU());
  auto gpu_device = Device::Get(_ctx);

  cpu_device->FreeDataSpace(CPU(), _cpu_hashtable);
  gpu_device->FreeDataSpace(_ctx, _gpu_hashtable);
}

void GPUCache::ExtractMissData(void *output, IdType *output_index,
                               size_t *num_output, const IdType *index,
                               const size_t num_index) {
  switch (_dtype) {
  case kF32:
    extract_miss_data<float>(_cpu_hashtable, output, output_index, num_output,
                             index, num_index, _all_data, _dim);
    break;
  case kF64:
    extract_miss_data<double>(_cpu_hashtable, output, output_index, num_output,
                              index, num_index, _all_data, _dim);
    break;
  case kF16:
    extract_miss_data<short>(_cpu_hashtable, output, output_index, num_output,
                             index, num_index, _all_data, _dim);
    break;
  case kU8:
    extract_miss_data<uint8_t>(_cpu_hashtable, output, output_index, num_output,
                               index, num_index, _all_data, _dim);
    break;
  case kI32:
    extract_miss_data<int32_t>(_cpu_hashtable, output, output_index, num_output,
                               index, num_index, _all_data, _dim);
    break;
  case kI64:
    extract_miss_data<int64_t>(_cpu_hashtable, output, output_index, num_output,
                               index, num_index, _all_data, _dim);
    break;
  default:
    CHECK(0);
  }
}

void GPUCache::CombineCacheData(void *output, const IdType *index,
                                const size_t num_index, StreamHandle stream) {
  CHECK_LE(num_index, _num_cached_nodes);

  auto device = Device::Get(_ctx);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  const uint32_t num_tiles =
      (num_index + Constant::kCudaTileSize - 1) / Constant::kCudaTileSize;

  const dim3 grid(num_tiles);
  const dim3 block(Constant::kCudaBlockSize);

  combine_cache_data<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(
          _gpu_hashtable, static_cast<char *>(output),
          static_cast<const char *>(_cache_gpu_data), index, num_index,
          GetTensorBytes(_dtype, {_dim}));

  device->StreamSync(_ctx, stream);
}

void GPUCache::CombineMissData(void *output, const IdType *index,
                               const size_t num_index, StreamHandle stream) {
  auto device = Device::Get(_ctx);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  device->StreamSync(_ctx, stream);
}

} // namespace cuda
} // namespace common
} // namespace samgraph