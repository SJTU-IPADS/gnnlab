#include <cassert>

#include <cub/cub.cuh>

#include "cuda_hashtable.h"
#include "logging.h"

namespace samgraph {
namespace common {
namespace cuda {

class MutableDeviceOrderedHashTable : public DeviceOrderedHashTable {
 public:
  typedef typename DeviceOrderedHashTable::Mapping *Iterator;

  explicit MutableDeviceOrderedHashTable(OrderedHashTable *const hostTable)
      : DeviceOrderedHashTable(hostTable->DeviceHandle()) {}

  inline __device__ Iterator Search(const IdType id) {
    const IdType pos = SearchForPosition(id);

    return GetMutable(pos);
  }

  inline __device__ bool AttemptInsertAt(const IdType pos, const IdType id,
                                         const IdType index) {
    const IdType key = atomicCAS(&GetMutable(pos)->key, Config::kEmptyKey, id);
    if (key == Config::kEmptyKey || key == id) {
      // we either set a match key, or found a matching key, so then place the
      // minimum index in position. Match the type of atomicMin, so ignore
      // linting
      atomicMin(&GetMutable(pos)->index, index); // NOLINT
      return true;
    } else {
      // we need to search elsewhere
      return false;
    }
  }

  inline __device__ Iterator Insert(const IdType id, const IdType index) {
    IdType pos = Hash(id);

    // linearly scan for an empty slot or matching entry
    IdType delta = 1;
    while (!AttemptInsertAt(pos, id, index)) {
      pos = Hash(pos + delta);
      delta += 1;
    }

    return GetMutable(pos);
  }

private:
  inline __device__ Iterator GetMutable(const IdType pos) {
    assert(pos < this->_size);
    // The parent class Device is read-only, but we ensure this can only be
    // constructed from a mutable version of OrderedHashTable, making this
    // a safe cast to perform.
    return const_cast<Iterator>(this->_table + pos);
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
                            MutableDeviceOrderedHashTable table) {
  assert(BLOCK_SIZE == blockDim.x);

  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_items) {
      table.Insert(items[index], index);
    }
  }
}

template <int BLOCK_SIZE, size_t TILE_SIZE>
__global__ void generate_hashmap_unique(const IdType *const items,
                                        const size_t num_items,
                                        MutableDeviceOrderedHashTable table) {
  assert(BLOCK_SIZE == blockDim.x);

  using Iterator = typename MutableDeviceOrderedHashTable::Iterator;

  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_items) {
      const Iterator pos = table.Insert(items[index], index);

      // since we are only inserting unique items, we know their local id
      // will be equal to their index
      pos->local = static_cast<IdType>(index);
    }
  }
}

template <int BLOCK_SIZE, size_t TILE_SIZE>
__global__ void count_hashmap(const IdType *items, const size_t num_items,
                              DeviceOrderedHashTable table,
                              IdType *const num_unique) {
  assert(BLOCK_SIZE == blockDim.x);

  using BlockReduce = typename cub::BlockReduce<IdType, BLOCK_SIZE>;
  using Mapping = typename DeviceOrderedHashTable::Mapping;

  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  IdType count = 0;

#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_items) {
      const Mapping &mapping = *table.Search(items[index]);
      if (mapping.index == index) {
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
                                IdType *const unique_items,
                                size_t *const num_unique_items) {
  assert(BLOCK_SIZE == blockDim.x);

  using FlagType = IdType;
  using BlockScan = typename cub::BlockScan<FlagType, BLOCK_SIZE>;
  using Mapping = typename DeviceOrderedHashTable::Mapping;

  constexpr const int32_t VALS_PER_THREAD = TILE_SIZE / BLOCK_SIZE;

  __shared__ typename BlockScan::TempStorage temp_space;

  const IdType offset = num_items_prefix[blockIdx.x];

  BlockPrefixCallbackOp<FlagType> prefix_op(0);

  // count successful placements
  for (int32_t i = 0; i < VALS_PER_THREAD; ++i) {
    const IdType index =
        threadIdx.x + i * BLOCK_SIZE + blockIdx.x * TILE_SIZE;

    FlagType flag;
    Mapping *kv;
    if (index < num_items) {
      kv = table.Search(items[index]);
      flag = kv->index == index;
    } else {
      flag = 0;
    }

    if (!flag) {
      kv = nullptr;
    }

    BlockScan(temp_space).ExclusiveSum(flag, flag, prefix_op);
    __syncthreads();

    if (kv) {
      const IdType pos = offset + flag;
      kv->local = pos;
      unique_items[pos] = items[index];
    }
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *num_unique_items = num_items_prefix[gridDim.x];
  }
}

// DeviceOrderedHashTable implementation
DeviceOrderedHashTable::DeviceOrderedHashTable(const Mapping *const table,
                                               const size_t size)
    : _table(table), _size(size) {}

DeviceOrderedHashTable OrderedHashTable::DeviceHandle() const {
  return DeviceOrderedHashTable(_table, _size);
}

// OrderedHashTable implementation
OrderedHashTable::OrderedHashTable(const size_t size, int device, cudaStream_t stream, const int scale)
    : _table(nullptr), _size(TableSize(size, scale)), _device(device) {
  // make sure we will at least as many buckets as items.
  CUDA_CALL(cudaMalloc(&_table, sizeof(Mapping) * _size));

  CUDA_CALL(cudaMemsetAsync(_table, Config::kEmptyKey,
                            sizeof(Mapping) * _size, stream));
}

OrderedHashTable::~OrderedHashTable() { CUDA_CALL(cudaFree(_table)); }

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
      <<<grid, block, 0, stream>>>(input, num_input, device_table);
  CUDA_CALL(cudaGetLastError());

  IdType *item_prefix;
  CUDA_CALL(cudaMalloc(&item_prefix, sizeof(IdType) * (num_input + 1)));

  count_hashmap<Config::kCudaBlockSize, Config::kCudaTileSize>
      <<<grid, block, 0, stream>>>(input, num_input, device_table, item_prefix);
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

  compact_hashmap<Config::kCudaBlockSize, Config::kCudaTileSize><<<grid, block, 0, stream>>>(
      input, num_input, device_table, item_prefix, unique, num_unique);
  CUDA_CALL(cudaGetLastError());
  CUDA_CALL(cudaFree(item_prefix));
}

void OrderedHashTable::FillWithUnique(const IdType *const input,
                                      const size_t num_input,
                                      cudaStream_t stream) {

  const size_t num_tiles = (num_input + Config::kCudaTileSize - 1) / Config::kCudaTileSize;

  const dim3 grid(num_tiles);
  const dim3 block(Config::kCudaBlockSize);

  auto device_table = MutableDeviceOrderedHashTable(this);

  generate_hashmap_unique<Config::kCudaBlockSize, Config::kCudaTileSize>
      <<<grid, block, 0, stream>>>(input, num_input, device_table);

  CUDA_CALL(cudaGetLastError());
}

} // namespace cuda
} // namespace common
} // namespace samgraph