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

#include <cuda_runtime.h>

#include <cub/cub.cuh>

#include "../common.h"
#include "../constant.h"
#include "../device.h"
#include "../function.h"
#include "../logging.h"
#include "../run_config.h"
#include "../profiler.h"
#include "../timer.h"
#include "collaborative_cache_manager.h"
#define SWITCH_TYPE(type, func)      \
  switch(type) {                     \
    case kF32: func(float);   break; \
    case kF64: func(double);  break; \
    case kF16: func(short);   break; \
    case kU8:  func(uint8_t); break; \
    case kI32: func(int32_t); break; \
    case kI64: func(int64_t); break; \
    default: CHECK(false);           \
  }


#define SAM_CUDA_PREPARE_1D(num_item) \
  const size_t num_tiles = RoundUpDiv((num_item), Constant::kCudaTileSize); \
  const dim3 grid(num_tiles); \
  const dim3 block(Constant::kCudaBlockSize);

namespace samgraph {
namespace common {
namespace dist {

namespace {

// using HashTableEntry = CollCacheManager::HashTableEntry;
using HashTableEntryLocation = CollCacheManager::HashTableEntryLocation;
using HashTableEntryOffset = CollCacheManager::HashTableEntryOffset;
using SrcKey = CollCacheManager::SrcKey;
using DstVal = CollCacheManager::DstVal;

template<typename LocationHolder>
struct LocationGetter {
__device__ static inline int location(const LocationHolder &s) { assert(false); return 0; }
};
template<>
struct LocationGetter<SrcKey> {
__device__ static inline int location(const SrcKey &s) { return s._location_id; }
};
template<>
struct LocationGetter<HashTableEntryLocation> {
__device__ static inline int location(const HashTableEntryLocation &s) { return s; }
};


template<typename SrcHolder, typename DstHolder>
struct OffsetGetter {
__device__ static inline IdType src(const SrcHolder *s, const DstHolder *d, const size_t idx) { assert(false); return 0; }
__device__ static inline IdType dst(const SrcHolder *s, const DstHolder *d, const size_t idx) { assert(false); return 0; }
};
template<>
struct OffsetGetter<SrcKey, DstVal> {
__device__ static inline IdType src(const SrcKey *, const DstVal *d, const size_t idx) { return d[idx]._src_offset; }
__device__ static inline IdType dst(const SrcKey *, const DstVal *d, const size_t idx) { return d[idx]._dst_offset; }
};
template<>
struct OffsetGetter<IdType, void> {
__device__ static inline IdType src(const IdType *s, const void *, const size_t idx) { return s[idx]; }
__device__ static inline IdType dst(const IdType *s, const void *, const size_t idx) { return idx; }
};


// template <typename T>
// __global__ void combine_miss_data(void *output, const void *miss,
//                                   const IdType *miss_dst_index,
//                                   const size_t num_miss, const size_t dim) {
//   T *output_data = reinterpret_cast<T *>(output);
//   const T *miss_data = reinterpret_cast<const T *>(miss);

//   size_t i = blockIdx.x * blockDim.y + threadIdx.y;
//   const size_t stride = blockDim.y * gridDim.x;

//   /** SXN: why need a loop?*/
//   /** SXN: ans: this loop is not necessary*/
//   while (i < num_miss) {
//     size_t col = threadIdx.x;
//     const size_t dst_idx = miss_dst_index[i];
//     while (col < dim) {
//       output_data[dst_idx * dim + col] = miss_data[i * dim + col];
//       col += blockDim.x;
//     }

//     i += stride;
//   }
// }

// template <typename T>
// __global__ void combine_cache_data(void *output, const IdType *cache_src_index,
//                                    const IdType *cache_dst_index,
//                                    const size_t num_cache, const void *cache,
//                                    size_t dim) {
//   T *output_data = reinterpret_cast<T *>(output);
//   const T *cache_data = reinterpret_cast<const T *>(cache);

//   size_t i = blockIdx.x * blockDim.y + threadIdx.y;
//   const size_t stride = blockDim.y * gridDim.x;

//   while (i < num_cache) {
//     size_t col = threadIdx.x;
//     const size_t src_idx = cache_src_index[i];
//     const size_t dst_idx = cache_dst_index[i];
//     while (col < dim) {
//       output_data[dst_idx * dim + col] = cache_data[src_idx * dim + col];
//       col += blockDim.x;
//     }
//     i += stride;
//   }
// }

template <typename T, typename SrcT, typename DstT, class Offset=OffsetGetter<SrcT, DstT>>
__global__ void combine_data(void *output,
                             const SrcT *src_index, const DstT *dst_index,
                             const size_t num_node,
                             const void *src_full,
                             size_t dim) {
  T *output_data = reinterpret_cast<T *>(output);
  const T *src_data = reinterpret_cast<const T *>(src_full);

  size_t i = blockIdx.x * blockDim.y + threadIdx.y;
  const size_t stride = blockDim.y * gridDim.x;

  while (i < num_node) {
    size_t col = threadIdx.x;
    const size_t src_offset = Offset::src(src_index, dst_index, i);
    const size_t dst_offset = Offset::dst(src_index, dst_index, i);
    while (col < dim) {
      output_data[dst_offset * dim + col] = src_data[src_offset * dim + col];
      col += blockDim.x;
    }
    i += stride;
  }
}

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void get_miss_cache_index(
    SrcKey* output_src_index, DstVal* output_dst_index,
    const IdType* nodes, const size_t num_nodes,
    const HashTableEntryLocation* hash_table_location,
    const HashTableEntryOffset* hash_table_offset) {

  assert(BLOCK_SIZE == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  for (size_t dst_idx = block_start + threadIdx.x; dst_idx < block_end;
       dst_idx += BLOCK_SIZE) {
    if (dst_idx < num_nodes) {
      const IdType node_id = nodes[dst_idx];
      output_src_index[dst_idx]._location_id = hash_table_location[node_id];
      output_dst_index[dst_idx]._src_offset = hash_table_offset[node_id];
      output_dst_index[dst_idx]._dst_offset = dst_idx;
      // output_src_index[dst_idx]._location_id = 0;
      // output_dst_index[dst_idx]._src_offset = 0;
      // output_dst_index[dst_idx]._dst_offset = 0;
    }
  }
}


template <size_t BLOCK_SIZE=Constant::kCudaBlockSize, size_t TILE_SIZE=Constant::kCudaTileSize, 
    typename Location_t, class LocGet = LocationGetter<Location_t>>
__global__ void find_boundary(
    const Location_t* output_src_index, const size_t len,
    IdType* boundary_list) {
  assert(BLOCK_SIZE == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  for (size_t src_offset = block_start + threadIdx.x; src_offset < block_end;
       src_offset += BLOCK_SIZE) {
    if (src_offset < len) {
      if (src_offset == len-1 || LocGet::location(output_src_index[src_offset]) != LocGet::location(output_src_index[src_offset+1])) {
        boundary_list[LocGet::location(output_src_index[src_offset])+1] = src_offset+1;
      } 
      // if (src_offset == 0 || output_src_index[src_offset]._location_id != output_src_index[src_offset-1]._location_id) {
      //   boundary_list[output_src_index[src_offset]._location_id] = src_offset;
      // }
    }
  }
}

template <size_t BLOCK_SIZE=Constant::kCudaBlockSize, size_t TILE_SIZE=Constant::kCudaTileSize>
__global__ void init_hash_table_cpu(
    HashTableEntryLocation* hash_table_location, HashTableEntryOffset* hash_table_offset, 
    const size_t num_total_nodes,
    const int cpu_location_id) {
  assert(BLOCK_SIZE == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  for (size_t node_id = block_start + threadIdx.x; node_id < block_end; node_id += BLOCK_SIZE) {
    if (node_id < num_total_nodes) {
      hash_table_location[node_id] = cpu_location_id;
      hash_table_offset[node_id] = node_id;
    }
  }
}


template <size_t BLOCK_SIZE=Constant::kCudaBlockSize, size_t TILE_SIZE=Constant::kCudaTileSize>
__global__ void init_hash_table_local(
    HashTableEntryLocation* hash_table_location, HashTableEntryOffset* hash_table_offset, 
    const IdType* local_nodes, const size_t num_node,
    const int local_location_id) {
  assert(BLOCK_SIZE == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  for (size_t offset = block_start + threadIdx.x; offset < block_end; offset += BLOCK_SIZE) {
    if (offset < num_node) {
      IdType node_id = local_nodes[offset];
      hash_table_location[node_id] = local_location_id;
      hash_table_offset[node_id] = offset;
    }
  }
}

template <size_t BLOCK_SIZE=Constant::kCudaBlockSize, size_t TILE_SIZE=Constant::kCudaTileSize>
__global__ void check_eq(const uint32_t * a, const uint32_t * b, const size_t n_elem) {

  assert(BLOCK_SIZE == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  for (size_t offset = block_start + threadIdx.x; offset < block_end; offset += BLOCK_SIZE) {
    if (offset < n_elem) {
      assert(a[offset] == b[offset]);
    }
  }
}

}  // namespace

CollCacheManager::CollCacheManager() {}
CollCacheManager::~CollCacheManager() {}

CollCacheManager::CollCacheManager(Context trainer_ctx, DataType dtype, size_t dim, int num_gpu)
    : _trainer_ctx(trainer_ctx),
      _dtype(dtype),
      _dim(dim),
      _num_location(num_gpu+1),
      _cpu_location_id(num_gpu) {
  LOG(INFO) << "Coll cache init with " << num_gpu << " gpus, " << _num_location << " locations";
}

void CollCacheManager::GetMissCacheIndex(
    SrcKey* & output_src_index, DstVal* & output_dst_index,
    const IdType* nodes, const size_t num_nodes, 
    StreamHandle stream) {
  auto cu_stream = static_cast<cudaStream_t>(stream);
  auto device = Device::Get(_trainer_ctx);

  output_src_index = static_cast<SrcKey*>(device->AllocWorkspace(_trainer_ctx, num_nodes * sizeof(SrcKey)));
  output_dst_index = static_cast<DstVal*>(device->AllocWorkspace(_trainer_ctx, num_nodes * sizeof(DstVal)));

  SrcKey * output_src_index_alter = static_cast<SrcKey*>(device->AllocWorkspace(_trainer_ctx, num_nodes * sizeof(SrcKey)));
  DstVal * output_dst_index_alter = static_cast<DstVal*>(device->AllocWorkspace(_trainer_ctx, num_nodes * sizeof(DstVal)));

  const size_t num_tiles = RoundUpDiv(num_nodes, Constant::kCudaTileSize);
  const dim3 grid(num_tiles);
  const dim3 block(Constant::kCudaBlockSize);
  LOG(DEBUG) << "CollCacheManager: GetMissCacheIndex - getting miss/hit index...";
  Timer t0;
  get_miss_cache_index<Constant::kCudaBlockSize, Constant::kCudaTileSize><<<grid, block, 0, cu_stream>>>(
    output_src_index, output_dst_index, nodes, num_nodes, this->_hash_table_location, this->_hash_table_offset);
  device->StreamSync(_trainer_ctx, stream);
  // std::cout << "coll get index "<< t0.Passed() << "\n";
  
  Timer t1;
  cub::DoubleBuffer<int> keys(reinterpret_cast<int*>(output_src_index), reinterpret_cast<int*>(output_src_index_alter));
  cub::DoubleBuffer<Id64Type> vals(reinterpret_cast<Id64Type*>(output_dst_index), reinterpret_cast<Id64Type*>(output_dst_index_alter));

  size_t workspace_bytes;
  LOG(DEBUG) << "CollCacheManager: GetMissCacheIndex - sorting according to group...";
  CUDA_CALL(cub::DeviceRadixSort::SortPairs(
      nullptr, workspace_bytes, keys, vals, num_nodes, 0, sizeof(SrcKey) * 8,
      cu_stream));

  void *workspace = device->AllocWorkspace(_trainer_ctx, workspace_bytes);
  CUDA_CALL(cub::DeviceRadixSort::SortPairs(
      workspace, workspace_bytes, keys, vals, num_nodes, 0, sizeof(SrcKey) * 8,
      cu_stream));
  device->StreamSync(_trainer_ctx, stream);
  LOG(DEBUG) << "CollCacheManager: GetMissCacheIndex - sorting according to group - done...";
  // std::cout << "coll sort index "<< t1.Passed() << "\n";

  Timer t2;
  output_src_index = reinterpret_cast<SrcKey*>(keys.Current());
  output_dst_index = reinterpret_cast<DstVal*>(vals.Current());

  device->FreeWorkspace(_trainer_ctx, keys.Alternate());
  device->FreeWorkspace(_trainer_ctx, vals.Alternate());
  device->FreeWorkspace(_trainer_ctx, workspace);
  // std::cout << "coll free workspace "<< t2.Passed() << "\n";
}

void CollCacheManager::SplitGroup(const SrcKey * src_index, const size_t len, IdType * & group_offset, StreamHandle stream){
  auto cu_stream = static_cast<cudaStream_t>(stream);
  auto device = Device::Get(_trainer_ctx);
  auto cpu_ctx = CPU(CPU_CUDA_HOST_MALLOC_DEVICE);
  const size_t num_tiles = RoundUpDiv(len, Constant::kCudaTileSize);
  const dim3 grid(num_tiles);
  const dim3 block(Constant::kCudaBlockSize);

  Timer t0;
  group_offset = (IdType*)Device::Get(cpu_ctx)->AllocWorkspace(cpu_ctx, sizeof(IdType) * (this->_num_location + 1));
  std::memset(group_offset, 0, sizeof(IdType) * (this->_num_location + 1));
  group_offset[this->_num_location] = len;
  LOG(DEBUG) << "CollCache: SplitGroup: legacy finding offset...";
  find_boundary<><<<grid, block, 0, cu_stream>>>(src_index, len, group_offset);
  device->StreamSync(_trainer_ctx, stream);
  LOG(DEBUG) << "CollCache: SplitGroup: legacy fixing offset...";
  for (int i = this->_num_location - 1; i > 0; i--) {
    if (group_offset[i] < group_offset[i-1]) {
      group_offset[i] = group_offset[i+1];
    }
  }
  // std::cout << "coll split group "<< t0.Passed() << "\n";
  LOG(DEBUG) << "CollCache: SplitGroup: legacy fixing done...";
}

namespace {
template <typename SrcT, typename DstT, class Offset = OffsetGetter<SrcT, DstT> >
void Combine(const SrcT * src_index, const DstT * dst_index,
    const size_t num_node, const void* src_data, void* output,
    Context _trainer_ctx, DataType _dtype, IdType _dim, StreamHandle stream);
}

void CollCacheManager::CombineOneGroup(const SrcKey * src_index, const DstVal * dst_index, const IdType* nodes, const size_t num_node, const void* src_data, void* output, StreamHandle stream) {
  Combine<SrcKey, DstVal, OffsetGetter<SrcKey, DstVal>>(src_index, dst_index, num_node, src_data, output, this->_trainer_ctx, this->_dtype, this->_dim, stream);
}
void CollCacheManager::CombineAllGroup(const SrcKey * src_index, const DstVal * dst_index, const IdType * group_offset, void* output, StreamHandle stream) {

}


namespace {
template <typename SrcT, typename DstT, class Offset>
void Combine(const SrcT * src_index, const DstT * dst_index, const size_t num_node, const void* src_data, void* output, Context _trainer_ctx, DataType _dtype, IdType _dim, StreamHandle stream) {
  if (num_node == 0) return;
  auto device = Device::Get(_trainer_ctx);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  dim3 block(256, 1);
  while (static_cast<size_t>(block.x) >= 2 * _dim) {
    block.x /= 2;
    block.y *= 2;
  }
  const dim3 grid(RoundUpDiv(num_node, static_cast<size_t>(block.y)));

  // #define FUNC(type) \
  // combine_data<type, src_getter, dst_getter><<<grid, block, 0, cu_stream>>>(output, src_index, dst_index, nodes, num_node, src_data, _dim);
  // SWITCH_TYPE(_dtype, FUNC);
  // #undef FUNC

  switch (_dtype) {
    case kF32:
      combine_data<float, SrcT, DstT, Offset><<<grid, block, 0, cu_stream>>>(
          output, src_index, dst_index, num_node, src_data, _dim);
      break;
    case kF64:
      combine_data<double, SrcT, DstT, Offset><<<grid, block, 0, cu_stream>>>(
          output, src_index, dst_index, num_node, src_data, _dim);
      break;
    case kF16:
      combine_data<short, SrcT, DstT, Offset><<<grid, block, 0, cu_stream>>>(
          output, src_index, dst_index, num_node, src_data, _dim);
      break;
    case kU8:
      combine_data<uint8_t, SrcT, DstT, Offset><<<grid, block, 0, cu_stream>>>(
          output, src_index, dst_index, num_node, src_data, _dim);
      break;
    case kI32:
      combine_data<int32_t, SrcT, DstT, Offset><<<grid, block, 0, cu_stream>>>(
          output, src_index, dst_index, num_node, src_data, _dim);
      break;
    case kI64:
      combine_data<int64_t, SrcT, DstT, Offset><<<grid, block, 0, cu_stream>>>(
          output, src_index, dst_index, num_node, src_data, _dim);
      break;
    default:
      CHECK(0);
  }

  device->StreamSync(_trainer_ctx, stream);
}
}

void CollCacheManager::ExtractFeat(const IdType* nodes, const size_t num_nodes,
                   void* output, StreamHandle stream, uint64_t task_key) {
  if (IsDirectMapping()) {
    // fast path
    // direct mapping from node id to freature, no need to go through hashtable
    LOG(DEBUG) << "CollCache: ExtractFeat: Direct mapping, going fast path... ";
    Timer t0;
    Combine<IdType, void, OffsetGetter<IdType, void>>(nodes, nullptr, num_nodes, _device_cache_data[0], output, this->_trainer_ctx, this->_dtype, this->_dim, stream);
    double combine_time = t0.Passed();
    if (task_key != 0xffffffffffffffff) {
      Profiler::Get().LogStep(task_key, kLogL1FeatureBytes, GetTensorBytes(_dtype, {num_nodes, _dim}));
      Profiler::Get().LogStep(task_key, kLogL1MissBytes, GetTensorBytes(_dtype, {num_nodes, _dim}));
      // Profiler::Get().LogStep(task_key, kLogL3CacheGetIndexTime, get_index_time);
      if (_cpu_location_id == -1) {
        // full cache
        Profiler::Get().LogStep(task_key, kLogL3CacheCombineCacheTime,combine_time);
      } else {
        // no cache
        Profiler::Get().LogStep(task_key, kLogL3CacheCombineMissTime,combine_time);
      }
      // Profiler::Get().LogStep(task_key, kLogL3CacheCombineCacheTime,combine_cache_time);
      Profiler::Get().LogEpochAdd(task_key, kLogEpochFeatureBytes,GetTensorBytes(_dtype, {num_nodes, _dim}));
      Profiler::Get().LogEpochAdd(task_key, kLogEpochMissBytes, GetTensorBytes(_dtype, {num_nodes, _dim}));
    }
  } else if (IsLegacy()) {
    auto trainer_gpu_device = Device::Get(_trainer_ctx);
    auto cpu_device = Device::Get(CPU(CPU_CUDA_HOST_MALLOC_DEVICE));
    SrcKey * src_index = nullptr;
    DstVal * dst_index = nullptr;
    LOG(DEBUG) << "CollCache: ExtractFeat: legacy, get miss cache index... ";
    Timer t0;
    GetMissCacheIndex(src_index, dst_index, nodes, num_nodes, stream);
    // std::cout << "Get Idx " << t0.Passed() << "\n";
    Timer t1;
    IdType * group_offset = nullptr;
    LOG(DEBUG) << "CollCache: ExtractFeat: legacy, splitting group... ";
    SplitGroup(src_index, num_nodes, group_offset, stream);
    double get_index_time = t0.Passed();
    // std::cout << "Split GrOup " <<t1.Passed() << "\n";
    double combine_times[2];
    for (int src_device_id = 1; src_device_id >= 0; src_device_id --) {
      LOG(DEBUG) << "CollCache: ExtractFeat: legacy, combining group " << src_device_id << " [" << group_offset[src_device_id] << "," << group_offset[src_device_id+1] << ")...";
      Timer t1;
      CombineOneGroup(
          src_index + group_offset[src_device_id], dst_index + group_offset[src_device_id], 
          nodes + group_offset[src_device_id], 
          group_offset[src_device_id+1] - group_offset[src_device_id], 
          _device_cache_data[src_device_id], output, stream);
      combine_times[src_device_id] = t1.Passed();
    }
    trainer_gpu_device->FreeWorkspace(_trainer_ctx, src_index);
    trainer_gpu_device->FreeWorkspace(_trainer_ctx, dst_index);
    if (task_key != 0xffffffffffffffff) {
      size_t num_miss = group_offset[2]- group_offset[1];
      // size_t num_hit = group_offset[1];
      Profiler::Get().LogStep(task_key, kLogL1FeatureBytes, GetTensorBytes(_dtype, {num_nodes, _dim}));
      Profiler::Get().LogStep(task_key, kLogL1MissBytes, GetTensorBytes(_dtype, {num_miss, _dim}));
      Profiler::Get().LogStep(task_key, kLogL3CacheGetIndexTime, get_index_time);
      Profiler::Get().LogStep(task_key, kLogL3CacheCombineMissTime,combine_times[1]);
      Profiler::Get().LogStep(task_key, kLogL3CacheCombineCacheTime,combine_times[0]);
      Profiler::Get().LogEpochAdd(task_key, kLogEpochFeatureBytes,GetTensorBytes(_dtype, {num_nodes, _dim}));
      Profiler::Get().LogEpochAdd(task_key, kLogEpochMissBytes, GetTensorBytes(_dtype, {num_miss, _dim}));
    }
    cpu_device->FreeWorkspace(CPU(CPU_CUDA_HOST_MALLOC_DEVICE), group_offset);
  } else {
    CHECK(false) << "Multi source extraction is not supported now";
  }
  // LOG(DEBUG) << "CollCache: check extract is same with no cache" << ")...";
  // size_t num_nbytes = num_nodes * GetDataTypeBytes(_dtype) * _dim;
  // void * naive_output = Device::Get(_trainer_ctx)->AllocWorkspace(_trainer_ctx, num_nbytes);
  // Combine<_src_direct, _dst_direct>(nullptr, nullptr, nodes, num_nodes, _device_cache_data[_cpu_location_id], naive_output, stream);
  // CheckCudaEqual(naive_output, output, num_nbytes);
  // Device::Get(_trainer_ctx)->FreeWorkspace(_trainer_ctx, naive_output);
}

CollCacheManager CollCacheManager::BuildLegacy(Context trainer_ctx,
                  void* cpu_src_data, DataType dtype, size_t dim,
                  TensorPtr cache_node_ptr, size_t num_total_nodes,
                  double cache_percentage, StreamHandle stream) {
  LOG(ERROR) << "Building Legacy Cache...";
  if (cache_percentage == 0 || cache_percentage == 1) {
    return BuildLegacy(trainer_ctx, cpu_src_data, dtype, dim, nullptr, num_total_nodes, cache_percentage, stream);
  }
  IdType* cache_node_list = (IdType*)cache_node_ptr->Data();
  CHECK_NE(cache_node_list, nullptr);
  return BuildLegacy(trainer_ctx, cpu_src_data, dtype, dim, cache_node_list, num_total_nodes, cache_percentage, stream);
}

CollCacheManager CollCacheManager::BuildLegacy(Context trainer_ctx,
                  void* cpu_src_data, DataType dtype, size_t dim,
                  const IdType* cache_node_list, size_t num_total_nodes,
                  double cache_percentage, StreamHandle stream) {
  if (cache_percentage == 0) {
    return BuildNoCache(trainer_ctx, cpu_src_data, dtype, dim, stream);
  } else if (cache_percentage == 1) {
    return BuildFullCache(trainer_ctx, cpu_src_data, dtype, dim, num_total_nodes, stream);
  }
  CHECK_NE(cache_node_list, nullptr);
  CollCacheManager cm(trainer_ctx, dtype, dim, 1);

  Timer t;

  size_t num_cached_nodes = num_total_nodes * cache_percentage;

  cm._cache_nbytes = GetTensorBytes(cm._dtype, {num_cached_nodes, cm._dim});

  // auto cpu_device = Device::Get(_extractor_ctx);
  auto trainer_gpu_device = Device::Get(trainer_ctx);

  // _cpu_hashtable = static_cast<IdType *>(
  //     cpu_device->AllocDataSpace(_extractor_ctx, sizeof(IdType) * _num_nodes));
  cm._device_cache_data.resize(2);
  cm._device_cache_data[0] = trainer_gpu_device->AllocDataSpace(trainer_ctx, cm._cache_nbytes);
  cm._device_cache_data[1] = cpu_src_data;
  cm._hash_table_location = (HashTableEntryLocation*)trainer_gpu_device->AllocDataSpace(trainer_ctx, sizeof(HashTableEntryLocation) * num_total_nodes);
  cm._hash_table_offset = (HashTableEntryOffset*)trainer_gpu_device->AllocDataSpace(trainer_ctx, sizeof(HashTableEntryOffset) * num_total_nodes);

  LOG(INFO) << "CollCacheManager: Initializing hashtable...";

  // 1. Initialize the hashtable with all miss
  auto cu_stream = static_cast<cudaStream_t>(stream);
  // auto cpu_ctx = CPU(CPU_CUDA_HOST_MALLOC_DEVICE);
  {
    SAM_CUDA_PREPARE_1D(num_total_nodes);
    init_hash_table_cpu<><<<grid, block, 0, cu_stream>>>(cm._hash_table_location, cm._hash_table_offset, num_total_nodes, cm._cpu_location_id);
    trainer_gpu_device->StreamSync(trainer_ctx, stream);
  }

  LOG(INFO) << "CollCacheManager: Initializing cache data...";
  // 2. use the hash table to extract cache from cpu
  SrcKey * src_index = nullptr;
  DstVal * dst_index = nullptr;
  {
    LOG(DEBUG) << "CollCacheManager: Initializing cache data - getting miss/hit index...";
    cm.GetMissCacheIndex(src_index, dst_index, cache_node_list, num_cached_nodes, stream);
    trainer_gpu_device->StreamSync(trainer_ctx, stream);
    // all location must be cpu now.
    LOG(DEBUG) << "CollCacheManager: Initializing cache data - getting cache data...";
    cm.CombineOneGroup(src_index, dst_index, cache_node_list, num_cached_nodes, cpu_src_data, cm._device_cache_data[0], stream);
    trainer_gpu_device->StreamSync(trainer_ctx, stream);
  }

  LOG(INFO) << "CollCacheManager: Add cache entry to hashtable...";
  // 3. modify hash table with cached nodes
  if (num_cached_nodes > 0){
    SAM_CUDA_PREPARE_1D(num_cached_nodes);
    init_hash_table_local<><<<grid, block, 0, cu_stream>>>(cm._hash_table_location, cm._hash_table_offset, cache_node_list, num_cached_nodes, 0);
    trainer_gpu_device->StreamSync(trainer_ctx, stream);
  }

  // 4. Free index
  trainer_gpu_device->FreeWorkspace(trainer_ctx, src_index);
  trainer_gpu_device->FreeWorkspace(trainer_ctx, dst_index);

  LOG(INFO) << "Collaborative GPU cache (policy: " << RunConfig::cache_policy
            << ") " << num_cached_nodes << " / " << num_total_nodes << " nodes ( "
            << ToPercentage(cache_percentage) << " | "
            << ToReadableSize(cm._cache_nbytes) << " | " << t.Passed()
            << " secs )";
  return cm;
}

CollCacheManager CollCacheManager::BuildNoCache(Context trainer_ctx,
                  void* cpu_src_data, DataType dtype, size_t dim,
                  StreamHandle stream) {
  CollCacheManager cm(trainer_ctx, dtype, dim, 0);
  cm._cpu_location_id = 0;

  Timer t;

  cm._cache_nbytes = 0;

  // auto cpu_device = Device::Get(_extractor_ctx);
  auto trainer_gpu_device = Device::Get(trainer_ctx);

  // _cpu_hashtable = static_cast<IdType *>(
  //     cpu_device->AllocDataSpace(_extractor_ctx, sizeof(IdType) * _num_nodes));
  cm._device_cache_data.resize(1);
  cm._device_cache_data[0] = cpu_src_data;
  cm._hash_table_location = nullptr;
  cm._hash_table_offset = nullptr;

  LOG(INFO) << "Collaborative GPU cache (policy: " << "no cache"
            << ") | " << t.Passed()
            << " secs )";
  return cm;
}

CollCacheManager CollCacheManager::BuildFullCache(Context trainer_ctx,
                  void* cpu_src_data, DataType dtype, size_t dim,
                  size_t num_total_nodes,
                  StreamHandle stream) {
  CollCacheManager cm(trainer_ctx, dtype, dim, 0);
  cm._cpu_location_id = -1;

  Timer t;

  cm._cache_nbytes = GetTensorBytes(cm._dtype, {num_total_nodes, cm._dim});

  auto trainer_gpu_device = Device::Get(trainer_ctx);

  void* local_cache = trainer_gpu_device->AllocDataSpace(trainer_ctx, cm._cache_nbytes);
  trainer_gpu_device->CopyDataFromTo(cpu_src_data, 0, local_cache, 0, cm._cache_nbytes, CPU(), trainer_ctx, stream);
  trainer_gpu_device->StreamSync(trainer_ctx, stream);

  cm._device_cache_data.resize(1);
  cm._device_cache_data[0] = local_cache;
  cm._hash_table_location = nullptr;
  cm._hash_table_offset = nullptr;

  LOG(INFO) << "Collaborative GPU cache (policy: " << "full cache"
            << ") " << num_total_nodes << " nodes ( "
            << ToReadableSize(cm._cache_nbytes) << " | " << t.Passed()
            << " secs )";
  return cm;
}

void CollCacheManager::CheckCudaEqual(const void * a, const void* b, const size_t nbytes, StreamHandle stream) {
  CHECK(nbytes % 4 == 0);
  const size_t n_elem = nbytes / 4;
  {
    SAM_CUDA_PREPARE_1D(n_elem);
    check_eq<><<<grid, block, 0, (cudaStream_t)stream>>>((const uint32_t*)a, (const uint32_t*)b, n_elem);
  }

  // {
  //   const size_t num_tiles = 1; // RoundUpDiv((n_elem), Constant::kCudaTileSize);
  //   const dim3 grid(num_tiles);
  //   const dim3 block(4);

  //   check_eq<4, 1000000><<<grid, block, 0, (cudaStream_t)stream>>>((const uint32_t*)a, (const uint32_t*)b, n_elem);
  // }
  CUDA_CALL(cudaStreamSynchronize((cudaStream_t)stream));
}


// void CollCacheManager::CombineMissData(void *output, const void *miss,
//                                       const IdType *miss_dst_index,
//                                       const size_t num_miss,
//                                       StreamHandle stream) {
//   LOG(DEBUG) << "CollCacheManager::CombineMissData():  num_miss " << num_miss;
//   if (num_miss == 0) return;

//   auto device = Device::Get(_trainer_ctx);
//   auto cu_stream = static_cast<cudaStream_t>(stream);

//   dim3 block(256, 1);
//   while (static_cast<size_t>(block.x) >= 2 * _dim) {
//     block.x /= 2;
//     block.y *= 2;
//   }
//   const dim3 grid(RoundUpDiv(num_miss, static_cast<size_t>(block.y)));

//   switch (_dtype) {
//     case kF32:
//       combine_miss_data<float><<<grid, block, 0, cu_stream>>>(
//           output, miss, miss_dst_index, num_miss, _dim);
//       break;
//     case kF64:
//       combine_miss_data<double><<<grid, block, 0, cu_stream>>>(
//           output, miss, miss_dst_index, num_miss, _dim);
//       break;
//     case kF16:
//       combine_miss_data<short><<<grid, block, 0, cu_stream>>>(
//           output, miss, miss_dst_index, num_miss, _dim);
//       break;
//     case kU8:
//       combine_miss_data<uint8_t><<<grid, block, 0, cu_stream>>>(
//           output, miss, miss_dst_index, num_miss, _dim);
//       break;
//     case kI32:
//       combine_miss_data<int32_t><<<grid, block, 0, cu_stream>>>(
//           output, miss, miss_dst_index, num_miss, _dim);
//       break;
//     case kI64:
//       combine_miss_data<int64_t><<<grid, block, 0, cu_stream>>>(
//           output, miss, miss_dst_index, num_miss, _dim);
//       break;
//     default:
//       CHECK(0);
//   }

//   device->StreamSync(_trainer_ctx, stream);
// }

// void CollCacheManager::CombineCacheData(void *output,
//                                        const IdType *cache_src_index,
//                                        const IdType *cache_dst_index,
//                                        const size_t num_cache,
//                                        StreamHandle stream) {
//   CHECK_LE(num_cache, _num_cached_nodes);

//   auto device = Device::Get(_trainer_ctx);
//   auto cu_stream = static_cast<cudaStream_t>(stream);

//   dim3 block(256, 1);
//   while (static_cast<size_t>(block.x) >= 2 * _dim) {
//     block.x /= 2;
//     block.y *= 2;
//   }
//   const dim3 grid(RoundUpDiv(num_cache, static_cast<size_t>(block.y)));

//   switch (_dtype) {
//     case kF32:
//       combine_cache_data<float><<<grid, block, 0, cu_stream>>>(
//           output, cache_src_index, cache_dst_index, num_cache,
//           _trainer_cache_data, _dim);
//       break;
//     case kF64:
//       combine_cache_data<double><<<grid, block, 0, cu_stream>>>(
//           output, cache_src_index, cache_dst_index, num_cache,
//           _trainer_cache_data, _dim);
//       break;
//     case kF16:
//       combine_cache_data<short><<<grid, block, 0, cu_stream>>>(
//           output, cache_src_index, cache_dst_index, num_cache,
//           _trainer_cache_data, _dim);
//       break;
//     case kU8:
//       combine_cache_data<uint8_t><<<grid, block, 0, cu_stream>>>(
//           output, cache_src_index, cache_dst_index, num_cache,
//           _trainer_cache_data, _dim);
//       break;
//     case kI32:
//       combine_cache_data<int32_t><<<grid, block, 0, cu_stream>>>(
//           output, cache_src_index, cache_dst_index, num_cache,
//           _trainer_cache_data, _dim);
//       break;
//     case kI64:
//       combine_cache_data<int64_t><<<grid, block, 0, cu_stream>>>(
//           output, cache_src_index, cache_dst_index, num_cache,
//           _trainer_cache_data, _dim);
//       break;
//     default:
//       CHECK(0);
//   }

//   device->StreamSync(_trainer_ctx, stream);
// }

}  // namespace dist
}  // namespace common
}  // namespace samgraph
