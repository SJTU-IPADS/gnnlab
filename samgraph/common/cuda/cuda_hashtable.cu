#include <cassert>
#include <cstdio>

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include "../common.h"
#include "../device.h"
#include "../logging.h"
#include "../timer.h"
#include "cuda_hashtable.h"

namespace samgraph {
namespace common {
namespace cuda {

class MutableDeviceOrderedHashTable : public DeviceOrderedHashTable {
public:
  typedef typename DeviceOrderedHashTable::Bucket *Iterator;
  typedef typename DeviceOrderedHashTable::Mapping *MapItem;

  explicit MutableDeviceOrderedHashTable(OrderedHashTable *const hostTable)
      : DeviceOrderedHashTable(hostTable->DeviceHandle()) {}

  inline __device__ Iterator Search(const IdType id) {
    const IdType pos = SearchForPosition(id);

    return GetMutable(pos);
  }

  inline __device__ bool AttemptInsertAt(const IdType pos, const IdType id,
                                         const IdType index,
                                         const IdType version) {
    const IdType key = atomicCAS(&GetMutable(pos)->key, Config::kEmptyKey, id);
    if (key == Config::kEmptyKey || key == id) {
      // we either set a match key, or found a matching key, so then place the
      // minimum index in position. Match the type of atomicMin, so ignore
      // linting
      atomicMin(&GetMutable(pos)->index, index);
      atomicCAS(&GetMutable(pos)->version, Config::kEmptyKey, version);
      return true;
    } else {
      // we need to search elsewhere
      return false;
    }
  }

  inline __device__ Iterator Insert(const IdType id, const IdType index,
                                    const IdType version) {
    IdType pos = Hash(id);

    // linearly scan for an empty slot or matching entry
    IdType delta = 1;
    while (!AttemptInsertAt(pos, id, index, version)) {
      pos = Hash(pos + delta);
      delta += 1;
    }

    return GetMutable(pos);
  }

  inline __device__ MapItem Map(const IdType pos, const IdType local) {
    GetMapItem(pos)->local = local;
    return GetMapItem(pos);
  }

private:
  inline __device__ Iterator GetMutable(const IdType pos) {
    assert(pos < this->_size);
    // The parent class Device is read-only, but we ensure this can only be
    // constructed from a mutable version of OrderedHashTable, making this
    // a safe cast to perform.
    return const_cast<Iterator>(this->_table + pos);
  }

  inline __device__ MapItem GetMapItem(const IdType pos) {
    assert(pos < this->_map_size);
    return const_cast<MapItem>(this->_map + pos);
  }
};

/**
 * Calculate the number of buckets in the hashtable. To guarantee we can
 * fill the hashtable in the worst case, we must use a number of buckets which
 * is a power of two.
 * https://en.wikipedia.org/wiki/Quadratic_probing#Limitations
 */
size_t TableSize(const size_t num, const size_t scale) {
  const size_t next_pow2 = 1 << static_cast<size_t>(1 + std::log2(num >> 1));
  return next_pow2 << scale;
}

/**
 * This structure is used with cub's block-level prefixscan in order to
 * keep a running sum as items are iteratively processed.
 */
template <typename T> struct BlockPrefixCallbackOp {
  T _running_total;

  __device__ BlockPrefixCallbackOp(const T running_total)
      : _running_total(running_total) {}

  __device__ T operator()(const T block_aggregate) {
    const T old_prefix = _running_total;
    _running_total += block_aggregate;
    return old_prefix;
  }
};

template <int BLOCK_SIZE, size_t TILE_SIZE>
__global__ void generate_hashmap_duplicates(const IdType *const items,
                                            const size_t num_items,
                                            MutableDeviceOrderedHashTable table,
                                            const IdType version) {
  assert(BLOCK_SIZE == blockDim.x);

  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_items) {
      table.Insert(items[index], index, version);
    }
  }
}

template <int BLOCK_SIZE, size_t TILE_SIZE>
__global__ void
generate_hashmap_unique(const IdType *const items, const size_t num_items,
                        MutableDeviceOrderedHashTable table,
                        const IdType global_offset, const IdType version) {
  assert(BLOCK_SIZE == blockDim.x);

  using Iterator = typename MutableDeviceOrderedHashTable::Iterator;

  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_items) {
      const Iterator bucket = table.Insert(items[index], index, version);
      IdType pos = global_offset + static_cast<IdType>(index);
      // since we are only inserting unique items, we know their local id
      // will be equal to their index
      bucket->local = pos;
      table.Map(pos, items[index]);
    }
  }
}

template <int BLOCK_SIZE, size_t TILE_SIZE>
__global__ void count_hashmap(const IdType *items, const size_t num_items,
                              DeviceOrderedHashTable table,
                              IdType *const num_unique, const IdType version) {
  assert(BLOCK_SIZE == blockDim.x);

  using BlockReduce = typename cub::BlockReduce<IdType, BLOCK_SIZE>;
  using Bucket = typename DeviceOrderedHashTable::Bucket;

  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  IdType count = 0;

#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_items) {
      const Bucket &bucket = *table.Search(items[index]);
      if (bucket.index == index && bucket.version == version) {
        ++count;
      }
    }
  }

  __shared__ typename BlockReduce::TempStorage temp_space;

  count = BlockReduce(temp_space).Sum(count);

  if (threadIdx.x == 0) {
    num_unique[blockIdx.x] = count;
    if (blockIdx.x == 0) {
      num_unique[gridDim.x] = 0;
    }
  }
}

template <int BLOCK_SIZE, size_t TILE_SIZE>
__global__ void
compact_hashmap(const IdType *const items, const size_t num_items,
                MutableDeviceOrderedHashTable table,
                const IdType *const num_items_prefix,
                size_t *const num_unique_items, const IdType global_offset,
                const IdType version) {
  assert(BLOCK_SIZE == blockDim.x);

  using FlagType = IdType;
  using BlockScan = typename cub::BlockScan<FlagType, BLOCK_SIZE>;
  using Bucket = typename DeviceOrderedHashTable::Bucket;

  constexpr const int32_t VALS_PER_THREAD = TILE_SIZE / BLOCK_SIZE;

  __shared__ typename BlockScan::TempStorage temp_space;

  const IdType offset = num_items_prefix[blockIdx.x];

  BlockPrefixCallbackOp<FlagType> prefix_op(0);

  // count successful placements
  for (int32_t i = 0; i < VALS_PER_THREAD; ++i) {
    const IdType index = threadIdx.x + i * BLOCK_SIZE + blockIdx.x * TILE_SIZE;

    FlagType flag;
    Bucket *kv;
    if (index < num_items) {
      kv = table.Search(items[index]);
      flag = kv->version == version && kv->index == index;
    } else {
      flag = 0;
    }

    if (!flag) {
      kv = nullptr;
    }

    BlockScan(temp_space).ExclusiveSum(flag, flag, prefix_op);
    __syncthreads();

    if (kv) {
      const IdType pos = global_offset + offset + flag;
      kv->local = pos;
      table.Map(pos, items[index]);
    }
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *num_unique_items = global_offset + num_items_prefix[gridDim.x];
  }
}

// DeviceOrderedHashTable implementation
DeviceOrderedHashTable::DeviceOrderedHashTable(const Bucket *const table,
                                               const Mapping *const map,
                                               const size_t size,
                                               const size_t mapping_size)
    : _table(table), _map(map), _size(size), _map_size(mapping_size) {}

DeviceOrderedHashTable OrderedHashTable::DeviceHandle() const {
  return DeviceOrderedHashTable(_table, _map, _size, _map_size);
}

// OrderedHashTable implementation
OrderedHashTable::OrderedHashTable(const size_t size, Context ctx,
                                   StreamHandle stream, const size_t scale)
    : _table(nullptr), _size(TableSize(size, scale)), _map_size(size),
      _ctx(ctx), _version(0), _offset(0) {
  // make sure we will at least as many buckets as items.
  auto device = Device::Get(_ctx);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  _table = static_cast<Bucket *>(
      device->AllocDataSpace(_ctx, sizeof(Bucket) * _size));
  _map = static_cast<Mapping *>(
      device->AllocDataSpace(_ctx, sizeof(Mapping) * _map_size));

  CUDA_CALL(cudaMemsetAsync(_table, (int)Config::kEmptyKey,
                            sizeof(Bucket) * _size, cu_stream));
  CUDA_CALL(cudaMemsetAsync(_map, (int)Config::kEmptyKey,
                            sizeof(Mapping) * _map_size, cu_stream));
  device->StreamSync(_ctx, stream);
}

OrderedHashTable::~OrderedHashTable() {
  Timer t;

  auto device = Device::Get(_ctx);
  device->FreeDataSpace(_ctx, _table);
  device->FreeDataSpace(_ctx, _map);

  LOG(DEBUG) << "free " << t.Passed();
}

void OrderedHashTable::Reset(StreamHandle stream) {
  auto cu_stream = static_cast<cudaStream_t>(stream);
  CUDA_CALL(cudaMemsetAsync(_table, (int)Config::kEmptyKey,
                            sizeof(Bucket) * _size, cu_stream));
  CUDA_CALL(cudaMemsetAsync(_map, (int)Config::kEmptyKey,
                            sizeof(Mapping) * _map_size, cu_stream));
  Device::Get(_ctx)->StreamSync(_ctx, stream);
  _version = 0;
  _offset = 0;
}

void OrderedHashTable::FillWithDuplicates(const IdType *const input,
                                          const size_t num_input,
                                          IdType *const unique,
                                          size_t *const num_unique,
                                          StreamHandle stream) {
  const size_t num_tiles =
      (num_input + Config::kCudaTileSize - 1) / Config::kCudaTileSize;

  const dim3 grid(num_tiles);
  const dim3 block(Config::kCudaBlockSize);

  auto device_table = MutableDeviceOrderedHashTable(this);
  auto device = Device::Get(_ctx);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  generate_hashmap_duplicates<Config::kCudaBlockSize, Config::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(input, num_input, device_table, _version);
  device->StreamSync(_ctx, stream);

  IdType *item_prefix = static_cast<IdType *>(
      device->AllocWorkspace(_ctx, sizeof(IdType) * (grid.x + 1)));
  LOG(DEBUG) << "OrderedHashTable::FillWithDuplicates cuda item_prefix malloc "
             << ToReadableSize(sizeof(IdType) * (grid.x + 1));

  count_hashmap<Config::kCudaBlockSize, Config::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(input, num_input, device_table,
                                      item_prefix, _version);
  device->StreamSync(_ctx, stream);

  size_t workspace_bytes;
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      nullptr, workspace_bytes, static_cast<IdType *>(nullptr),
      static_cast<IdType *>(nullptr), grid.x + 1, cu_stream));
  device->StreamSync(_ctx, stream);

  void *workspace = device->AllocWorkspace(_ctx, workspace_bytes);
  LOG(DEBUG) << "OrderedHashTable::FillWithDuplicates cuda item_prefix malloc "
             << ToReadableSize(sizeof(IdType) * (num_input + 1));

  CUDA_CALL(cub::DeviceScan::ExclusiveSum(workspace, workspace_bytes,
                                          item_prefix, item_prefix, grid.x + 1,
                                          cu_stream));
  device->StreamSync(_ctx, stream);

  size_t *gpu_num_unique =
      static_cast<size_t *>(device->AllocWorkspace(_ctx, sizeof(size_t)));
  LOG(DEBUG)
      << "OrderedHashTable::FillWithDuplicates cuda gpu_num_unique malloc "
      << ToReadableSize(sizeof(size_t));

  compact_hashmap<Config::kCudaBlockSize, Config::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(input, num_input, device_table,
                                      item_prefix, gpu_num_unique, _offset,
                                      _version);
  device->StreamSync(_ctx, stream);

  device->CopyDataFromTo(gpu_num_unique, 0, num_unique, 0, sizeof(size_t), _ctx,
                         CPU(), stream);
  device->StreamSync(_ctx, stream);

  LOG(DEBUG) << "OrderedHashTable::FillWithDuplicates num_unique "
             << *num_unique;

  device->CopyDataFromTo(_map, 0, unique, 0, sizeof(IdType) * (*num_unique),
                         _ctx, _ctx, stream);
  device->StreamSync(_ctx, stream);

  device->FreeWorkspace(_ctx, gpu_num_unique);
  device->FreeWorkspace(_ctx, item_prefix);
  device->FreeWorkspace(_ctx, workspace);

  _version++;
  _offset = *num_unique;
}

void OrderedHashTable::FillWithUnique(const IdType *const input,
                                      const size_t num_input,
                                      StreamHandle stream) {

  const size_t num_tiles =
      (num_input + Config::kCudaTileSize - 1) / Config::kCudaTileSize;

  const dim3 grid(num_tiles);
  const dim3 block(Config::kCudaBlockSize);

  auto device_table = MutableDeviceOrderedHashTable(this);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  generate_hashmap_unique<Config::kCudaBlockSize, Config::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(input, num_input, device_table, _offset,
                                      _version);
  Device::Get(_ctx)->StreamSync(_ctx, stream);

  _version++;
  _offset += num_input;

  LOG(DEBUG) << "OrderedHashTable::FillWithUnique insert " << num_input
             << " items, now " << _offset << " in total";
}

} // namespace cuda
} // namespace common
} // namespace samgraph