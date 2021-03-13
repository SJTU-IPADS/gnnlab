#include <cassert>

#include <cub/cub.cuh>

#include "cuda_hashtable.h"
#include "logging.h"

namespace samgraph {
namespace common {
namespace cuda {

class MutableDeviceOrderedHashTable : public DeviceOrderedHashTable {
 public:
  typedef typename DeviceOrderedHashTable::Bucket *Iterator;
  typedef typename DeviceOrderedHashTable::Mapping *MappingPtr;

  explicit MutableDeviceOrderedHashTable(OrderedHashTable *const hostTable)
      : DeviceOrderedHashTable(hostTable->DeviceHandle()) {}

  inline __device__ Iterator Search(const IdType id) {
    const IdType pos = SearchForPosition(id);

    return GetMutable(pos);
  }

  inline __device__ bool AttemptInsertAt(const IdType pos, const IdType id,
                                         const IdType index, const IdType version) {
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

  inline __device__ Iterator Insert(const IdType id, const IdType index, const IdType version) {
    IdType pos = Hash(id);

    // linearly scan for an empty slot or matching entry
    IdType delta = 1;
    while (!AttemptInsertAt(pos, id, index, version)) {
      pos = Hash(pos + delta);
      delta += 1;
    }

    return GetMutable(pos);
  }

  inline __device__ MappingPtr Map(const IdType pos, const IdType local) {
    GetMapping(pos)->local = local;
    return GetMapping(pos);
  }

private:
  inline __device__ Iterator GetMutable(const IdType pos) {
    assert(pos < this->_size);
    // The parent class Device is read-only, but we ensure this can only be
    // constructed from a mutable version of OrderedHashTable, making this
    // a safe cast to perform.
    return const_cast<Iterator>(this->_table + pos);
  }

  inline __device__ MappingPtr GetMapping(const IdType pos) {
    assert(pos < this->_mapping_size);
    return const_cast<MappingPtr>(this->_mapping + pos);
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
template <typename T>
struct BlockPrefixCallbackOp {
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
__global__ void
generate_hashmap_duplicates(const IdType *const items,
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
__global__ void generate_hashmap_unique(const IdType *const items,
                                        const size_t num_items,
                                        MutableDeviceOrderedHashTable table,
                                        const IdType global_offset,
                                        const IdType version) {
  assert(BLOCK_SIZE == blockDim.x);

  using Iterator = typename MutableDeviceOrderedHashTable::Iterator;

  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_items) {
      const Iterator pos = table.Insert(items[index], index, version);

      // since we are only inserting unique items, we know their local id
      // will be equal to their index
      pos->local = global_offset + static_cast<IdType>(index);
    }
  }
}

template <int BLOCK_SIZE, size_t TILE_SIZE>
__global__ void count_hashmap(const IdType *items, const size_t num_items,
                              DeviceOrderedHashTable table,
                              IdType *const num_unique,
                              const IdType version) {
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
__global__ void compact_hashmap(const IdType *const items,
                                const size_t num_items,
                                MutableDeviceOrderedHashTable table,
                                const IdType *const num_items_prefix,
                                size_t *const num_unique_items,
                                const IdType global_offset,
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
    const IdType index =
        threadIdx.x + i * BLOCK_SIZE + blockIdx.x * TILE_SIZE;

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
                                               const Mapping *const mapping,
                                               const size_t size,
                                               const size_t mapping_size)
    : _table(table), _mapping(mapping), _size(size), _mapping_size(mapping_size) {}

DeviceOrderedHashTable OrderedHashTable::DeviceHandle() const {
  return DeviceOrderedHashTable(_table, _mapping, _size, _mapping_size);
}

// OrderedHashTable implementation
OrderedHashTable::OrderedHashTable(const size_t size, int device, cudaStream_t stream, const size_t scale)
    : _table(nullptr), _size(TableSize(size, scale)), _mapping_size(size), _device(device), _version(0), _offset(0) {
  // make sure we will at least as many buckets as items.
  CUDA_CALL(cudaMalloc(&_table, sizeof(Bucket) * _size));
  CUDA_CALL(cudaMalloc(&_mapping, sizeof(Mapping) * _mapping_size));

  CUDA_CALL(cudaMemsetAsync(_table, (int)Config::kEmptyKey,
                            sizeof(Bucket) * _size, stream));
  CUDA_CALL(cudaMemsetAsync(_mapping, (int)Config::kEmptyKey,
                            sizeof(Mapping) *_mapping_size, stream));
}

OrderedHashTable::~OrderedHashTable() { 
  CUDA_CALL(cudaFree(_table));
  CUDA_CALL(cudaFree(_mapping));
}

void OrderedHashTable::FillWithDuplicates(const IdType *const input,
                                          const size_t num_input,
                                          IdType *const unique,
                                          size_t *const num_unique,
                                          cudaStream_t stream) {
  const size_t num_tiles = (num_input + Config::kCudaTileSize - 1) / Config::kCudaTileSize;

  const dim3 grid(num_tiles);
  const dim3 block(Config::kCudaBlockSize);

  auto device_table = MutableDeviceOrderedHashTable(this);

  generate_hashmap_duplicates<Config::kCudaBlockSize, Config::kCudaTileSize>
      <<<grid, block, 0, stream>>>(input, num_input, device_table, _version);
  CUDA_CALL(cudaGetLastError());

  IdType *item_prefix;
  CUDA_CALL(cudaMalloc(&item_prefix, sizeof(IdType) * (num_input + 1)));

  count_hashmap<Config::kCudaBlockSize, Config::kCudaTileSize>
      <<<grid, block, 0, stream>>>(input, num_input, device_table, item_prefix, _version);
  CUDA_CALL(cudaGetLastError());

  size_t workspace_bytes;
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      nullptr, workspace_bytes, static_cast<IdType *>(nullptr),
      static_cast<IdType *>(nullptr), grid.x + 1, stream));
  void *workspace;
  CUDA_CALL(cudaMalloc(&workspace, workspace_bytes));

  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      workspace, workspace_bytes, item_prefix, item_prefix, grid.x + 1, stream));

  CUDA_CALL(cudaFree(workspace));

  size_t *d_num_unique;
  CUDA_CALL(cudaMalloc(&d_num_unique, sizeof(size_t)));
  compact_hashmap<Config::kCudaBlockSize, Config::kCudaTileSize><<<grid, block, 0, stream>>>(
      input, num_input, device_table, item_prefix, d_num_unique, _offset, _version);
  CUDA_CALL(cudaGetLastError());
  CUDA_CALL(cudaFree(item_prefix));

  CUDA_CALL(cudaMemcpyAsync(num_unique, d_num_unique, sizeof(size_t), cudaMemcpyDeviceToHost, stream));
  CUDA_CALL(cudaFree(d_num_unique));

  CUDA_CALL(cudaMemcpyAsync(unique, _mapping, sizeof(IdType) * (*num_unique), cudaMemcpyDeviceToDevice, stream));
  CUDA_CALL(cudaStreamSynchronize(stream));

  _version++;
  _offset = *num_unique;
}

void OrderedHashTable::FillWithUnique(const IdType *const input,
                                      const size_t num_input,
                                      cudaStream_t stream) {

  const size_t num_tiles = (num_input + Config::kCudaTileSize - 1) / Config::kCudaTileSize;

  const dim3 grid(num_tiles);
  const dim3 block(Config::kCudaBlockSize);

  auto device_table = MutableDeviceOrderedHashTable(this);

  generate_hashmap_unique<Config::kCudaBlockSize, Config::kCudaTileSize>
      <<<grid, block, 0, stream>>>(input, num_input, device_table, _offset, _version);

  CUDA_CALL(cudaGetLastError());
}

} // namespace cuda
} // namespace common
} // namespace samgraph