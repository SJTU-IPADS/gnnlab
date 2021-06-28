
#include <string>
#include <unordered_map>

#include "../common.h"
#include "../constant.h"
#include "../device.h"
#include "../function.h"
#include "../logging.h"
#include "../run_config.h"
#include "../timer.h"
#include "cuda_cache_manager.h"

namespace samgraph {
namespace common {
namespace cuda {

namespace {

template <typename T>
void extract_miss_data(void *output_miss, const IdType *miss_src_index,
                       const size_t num_miss, const void *src, size_t dim) {
  auto cpu_device = Device::Get(CPU());

  T *output_miss_data = reinterpret_cast<T *>(output_miss);
  const T *cpu_src_data = reinterpret_cast<const T *>(src);

#pragma omp parallel for num_threads(RunConfig::kOMPThreadNum)
  for (size_t i = 0; i < num_miss; i++) {
#pragma omp simd
    for (size_t j = 0; j < dim; j++) {
      output_miss_data[i * dim + j] = cpu_src_data[miss_src_index[i] * dim + j];
    }
  }
}

}  // namespace

GPUCacheManager::GPUCacheManager(Context sampler_ctx, Context trainer_ctx,
                                 const void *cpu_src_data, DataType dtype,
                                 size_t dim, const IdType *nodes,
                                 size_t num_nodes, double cache_percentage)
    : _sampler_ctx(sampler_ctx),
      _trainer_ctx(trainer_ctx),
      _cache_percentage(cache_percentage),
      _num_nodes(num_nodes),
      _num_cached_nodes(num_nodes * cache_percentage),
      _dtype(dtype),
      _dim(dim),
      _cpu_src_data(cpu_src_data) {
  Timer t;

  _cache_nbytes = GetTensorBytes(_dtype, {_num_cached_nodes, _dim});

  auto cpu_device = Device::Get(CPU());
  auto sampler_gpu_device = Device::Get(_sampler_ctx);
  auto trainer_gpu_device = Device::Get(_trainer_ctx);

  IdType *tmp_cpu_hashtable = static_cast<IdType *>(
      cpu_device->AllocDataSpace(CPU(), sizeof(IdType) * _num_nodes));
  _sampler_gpu_hashtable =
      static_cast<IdType *>(sampler_gpu_device->AllocDataSpace(
          _sampler_ctx, sizeof(IdType) * _num_nodes));

  void *tmp_cpu_data = cpu_device->AllocDataSpace(CPU(), _cache_nbytes);
  _trainer_cache_data =
      trainer_gpu_device->AllocDataSpace(_trainer_ctx, _cache_nbytes);

  // 1. Initialize the cpu hashtable
#pragma omp parallel for num_threads(RunConfig::kOMPThreadNum)
  for (size_t i = 0; i < _num_nodes; i++) {
    tmp_cpu_hashtable[i] = Constant::kEmptyKey;
  }

  // 2. Populate the cpu hashtable
#pragma omp parallel for num_threads(RunConfig::kOMPThreadNum)
  for (size_t i = 0; i < _num_cached_nodes; i++) {
    tmp_cpu_hashtable[nodes[i]] = i;
  }

  // 3. Populate the cache in cpu memory
  cpu::CPUExtract(tmp_cpu_data, _cpu_src_data, nodes, _num_cached_nodes, _dim,
                  _dtype);

  // 4. Copy the cache from the cpu memory to gpu memory
  sampler_gpu_device->CopyDataFromTo(
      tmp_cpu_hashtable, 0, _sampler_gpu_hashtable, 0,
      sizeof(IdType) * _num_nodes, CPU(), _sampler_ctx);
  trainer_gpu_device->CopyDataFromTo(tmp_cpu_data, 0, _trainer_cache_data, 0,
                                     _cache_nbytes, CPU(), _trainer_ctx);

  // 5. Free the cpu tmp cache data
  cpu_device->FreeDataSpace(CPU(), tmp_cpu_hashtable);
  cpu_device->FreeDataSpace(CPU(), tmp_cpu_data);

  std::unordered_map<CachePolicy, std::string> policy2str = {
      {kCacheByDegree, "degree"}, {kCacheByHeuristic, "heuristic"}};

  LOG(INFO) << "GPU cache (policy: " << policy2str.at(RunConfig::cache_policy)
            << ") " << _num_cached_nodes << " / " << _num_nodes << " nodes ( "
            << ToPercentage(_cache_percentage) << " | "
            << ToReadableSize(_cache_nbytes) << " | " << t.Passed()
            << " secs )";
}

GPUCacheManager::~GPUCacheManager() {
  auto sampler_device = Device::Get(_sampler_ctx);
  auto trainer_device = Device::Get(_trainer_ctx);

  sampler_device->FreeDataSpace(_sampler_ctx, _sampler_gpu_hashtable);
  trainer_device->FreeDataSpace(_trainer_ctx, _trainer_cache_data);
}

void GPUCacheManager::ExtractMissData(void *output_miss,
                                      const IdType *miss_src_index,
                                      const size_t num_miss) {
  switch (_dtype) {
    case kF32:
      extract_miss_data<float>(output_miss, miss_src_index, num_miss,
                               _cpu_src_data, _dim);
      break;
    case kF64:
      extract_miss_data<double>(output_miss, miss_src_index, num_miss,
                                _cpu_src_data, _dim);
      break;
    case kF16:
      extract_miss_data<short>(output_miss, miss_src_index, num_miss,
                               _cpu_src_data, _dim);
      break;
    case kU8:
      extract_miss_data<uint8_t>(output_miss, miss_src_index, num_miss,
                                 _cpu_src_data, _dim);
      break;
    case kI32:
      extract_miss_data<int32_t>(output_miss, miss_src_index, num_miss,
                                 _cpu_src_data, _dim);
      break;
    case kI64:
      extract_miss_data<int64_t>(output_miss, miss_src_index, num_miss,
                                 _cpu_src_data, _dim);
      break;
    default:
      CHECK(0);
  }
}

}  // namespace cuda
}  // namespace common
}  // namespace samgraph