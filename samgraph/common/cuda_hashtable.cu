#include <cassert>

#include "cub/cub.cuh"

#include "cuda_hashtable.h"
#include "logging.h"

namespace {

/**
 * @brief This is the mutable version of the DeviceOrderedHashTable, for use in
 * inserting elements into the hashtable.
 *
 */
class MutableDeviceOrderedHashTable : public DeviceOrderedHashTable {
 public:
  typedef typename DeviceOrderedHashTable::Mapping *Iterator;
  static constexpr uint32_t kEmptyKey = DeviceOrderedHashTable::kEmptyKey;

  /**
   * @brief Create a new mutable hashtable for use on the device.
   *
   * @param hostTable The original hash table on the host.
   */
  explicit MutableDeviceOrderedHashTable(OrderedHashTable *const hostTable)
      : DeviceOrderedHashTable(hostTable->DeviceHandle()) {}

  /**
   * @brief Find the mutable mapping of a given key within the hash table.
   *
   * WARNING: The key must exist within the hashtable. Searching for a key not
   * in the hashtable is undefined behavior.
   *
   * @param id The key to search for.
   *
   * @return The mapping.
   */
  inline __device__ Iterator Search(const uint32_t id) {
    const uint32_t pos = SearchForPosition(id);

    return GetMutable(pos);
  }

  /**
   * \brief Attempt to insert into the hash table at a specific location.
   *
   * \param pos The position to insert at.
   * \param id The ID to insert into the hash table.
   * \param index The original index of the item being inserted.
   *
   * \return True, if the insertion was successful.
   */
  inline __device__ bool AttemptInsertAt(const size_t pos, const uint32_t id,
                                         const size_t index) {
    const uint32_t key = atomicCAS(&GetMutable(pos)->key, kEmptyKey, id);
    if (key == kEmptyKey || key == id) {
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

  /**
   * @brief Insert key-index pair into the hashtable.
   *
   * @param id The ID to insert.
   * @param index The index at which the ID occured.
   *
   * @return An iterator to inserted mapping.
   */
  inline __device__ Iterator Insert(const uint32_t id, const size_t index) {
    size_t pos = Hash(id);

    // linearly scan for an empty slot or matching entry
    uint32_t delta = 1;
    while (!AttemptInsertAt(pos, id, index)) {
      pos = Hash(pos + delta);
      delta += 1;
    }

    return GetMutable(pos);
  }

private:
  /**
   * @brief Get a mutable iterator to the given bucket in the hashtable.
   *
   * @param pos The given bucket.
   *
   * @return The iterator.
   */
  inline __device__ Iterator GetMutable(const size_t pos) {
    assert(pos < this->size_);
    // The parent class Device is read-only, but we ensure this can only be
    // constructed from a mutable version of OrderedHashTable, making this
    // a safe cast to perform.
    return const_cast<Iterator>(this->table_ + pos);
  }
};

/**
 * @brief Calculate the number of buckets in the hashtable. To guarantee we can
 * fill the hashtable in the worst case, we must use a number of buckets which
 * is a power of two.
 * https://en.wikipedia.org/wiki/Quadratic_probing#Limitations
 *
 * @param num The number of items to insert (should be an upper bound on the
 * number of unique keys).
 * @param scale The power of two larger the number of buckets should be than the
 * unique keys.
 *
 * @return The number of buckets the table should contain.
 */
size_t TableSize(const size_t num, const int scale) {
  const size_t next_pow2 = 1 << static_cast<size_t>(1 + std::log2(num >> 1));
  return next_pow2 << scale;
}

/**
 * @brief This structure is used with cub's block-level prefixscan in order to
 * keep a running sum as items are iteratively processed.
 *
 * @tparam IdType The type to perform the prefixsum on.
 */

template <typename IdType> struct BlockPrefixCallbackOp {
  uint32_t running_total_;

  __device__ BlockPrefixCallbackOp(const IdType running_total)
      : running_total_(running_total) {}

  __device__ IdType operator()(const IdType block_aggregate) {
    const IdType old_prefix = running_total_;
    running_total_ += block_aggregate;
    return old_prefix;
  }
};

} // namespace

/**
 * \brief This generates a hash map where the keys are the global item numbers,
 * and the values are indexes, and inputs may have duplciates.
 *
 * \tparam BLOCK_SIZE The size of the thread block.
 * \tparam TILE_SIZE The number of entries each thread block will process.
 * \param items The items to insert.
 * \param num_items The number of items to insert.
 * \param table The hash table.
 */
template <int BLOCK_SIZE, size_t TILE_SIZE>
__global__ void
generate_hashmap_duplicates(const uint32_t *const items,
                            const uint32_t num_items,
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

/**
 * \brief This generates a hash map where the keys are the global item numbers,
 * and the values are indexes, and all inputs are unique.
 *
 * \tparam BLOCK_SIZE The size of the thread block.
 * \tparam TILE_SIZE The number of entries each thread block will process.
 * \param items The unique items to insert.
 * \param num_items The number of items to insert.
 * \param table The hash table.
 */
template <int BLOCK_SIZE, size_t TILE_SIZE>
__global__ void generate_hashmap_unique(const uint32_t *const items,
                                        const uint32_t num_items,
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
      pos->local = static_cast<uint32_t>(index);
    }
  }
}

/**
 * \brief This counts the number of nodes inserted per thread block.
 *
 * \tparam BLOCK_SIZE The size of the thread block.
 * \tparam TILE_SIZE The number of entries each thread block will process.
 * \param input The nodes to insert.
 * \param num_input The number of nodes to insert.
 * \param table The hash table.
 * \param num_unique The number of nodes inserted into the hash table per thread
 * block.
 */
template <int BLOCK_SIZE, size_t TILE_SIZE>
__global__ void count_hashmap(const uint32_t *items, const size_t num_items,
                              DeviceOrderedHashTable table,
                              uint32_t *const num_unique) {
  assert(BLOCK_SIZE == blockDim.x);

  using BlockReduce = typename cub::BlockReduce<uint32_t, BLOCK_SIZE>;
  using Mapping = typename DeviceOrderedHashTable::Mapping;

  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  uint32_t count = 0;

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

/**
 * \brief Update the local numbering of elements in the hashmap.
 *
 * \tparam BLOCK_SIZE The size of the thread blocks.
 * \tparam TILE_SIZE The number of elements each thread block works on.
 * \param items The set of non-unique items to update from.
 * \param num_items The number of non-unique items.
 * \param table The hash table.
 * \param num_items_prefix The number of unique items preceding each thread
 * block.
 * \param unique_items The set of unique items (output).
 * \param num_unique_items The number of unique items (output).
 */
template <int BLOCK_SIZE, size_t TILE_SIZE>
__global__ void compact_hashmap(const uint32_t *const items,
                                const size_t num_items,
                                MutableDeviceOrderedHashTable table,
                                const uint32_t *const num_items_prefix,
                                uint32_t *const unique_items,
                                uint32_t *const num_unique_items) {
  assert(BLOCK_SIZE == blockDim.x);

  using FlagType = uint16_t;
  using BlockScan = typename cub::BlockScan<FlagType, BLOCK_SIZE>;
  using Mapping = typename DeviceOrderedHashTable::Mapping;

  constexpr const int32_t VALS_PER_THREAD = TILE_SIZE / BLOCK_SIZE;

  __shared__ typename BlockScan::TempStorage temp_space;

  const uint32_t offset = num_items_prefix[blockIdx.x];

  BlockPrefixCallbackOp<FlagType> prefix_op(0);

  // count successful placements
  for (int32_t i = 0; i < VALS_PER_THREAD; ++i) {
    const uint32_t index =
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
      const uint32_t pos = offset + flag;
      kv->local = pos;
      unique_items[pos] = items[index];
    }
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *num_unique_items = num_items_prefix[gridDim.x];
  }
}

namespace samgraph {
namespace common {
namespace cuda {

// DeviceOrderedHashTable implementation
DeviceOrderedHashTable::DeviceOrderedHashTable(const Mapping *const table,
                                               const size_t size)
    : table_(table), size_(size) {}

DeviceOrderedHashTable OrderedHashTable::DeviceHandle() const {
  return DeviceOrderedHashTable(table_, size_);
}

// OrderedHashTable implementation

OrderedHashTable::OrderedHashTable(const size_t size, const int scale)
    : table_(nullptr), size_(TableSize(size, scale)) {
  // make sure we will at least as many buckets as items.
  CUDA_CALL(cudaMalloc(&table_, sizeof(Mapping) * size_));

  CUDA_CALL(cudaMemset(table_, DeviceOrderedHashTable::kEmptyKey,
                        sizeof(Mapping) * size_));
}

OrderedHashTable::~OrderedHashTable() { CUDA_CALL(cudaFree(table_)); }

void OrderedHashTable::FillWithDuplicates(const uint32_t *const input,
                                          const size_t num_input,
                                          uint32_t *const unique,
                                          uint32_t *const num_unique,
                                          cudaStream_t stream) {
  const uint32_t num_tiles = (num_input + SAM_CUDA_TILE_SIZE - 1) / SAM_CUDA_TILE_SIZE;

  const dim3 grid(num_tiles);
  const dim3 block(SAM_CUDA_BLOCK_SIZE);

  auto device_table = MutableDeviceOrderedHashTable(this);

  generate_hashmap_duplicates<SAM_CUDA_BLOCK_SIZE, SAM_CUDA_TILE_SIZE>
      <<<grid, block, 0, stream>>>(input, num_input, device_table);
  CUDA_CALL(cudaGetLastError());

  uint32_t *item_prefix;
  CUDA_CALL(cudaMalloc(&item_prefix, sizeof(uint32_t) * (num_input + 1)));

  count_hashmap<SAM_CUDA_BLOCK_SIZE, SAM_CUDA_TILE_SIZE>
      <<<grid, block, 0, stream>>>(input, num_input, device_table, item_prefix);
  CUDA_CALL(cudaGetLastError());

  size_t workspace_bytes;
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      nullptr, workspace_bytes, static_cast<uint32_t *>(nullptr),
      static_cast<uint32_t *>(nullptr), grid.x + 1));
  void *workspace;
  CUDA_CALL(cudaMalloc(&workspace, workspace_bytes));

  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      workspace, workspace_bytes, item_prefix, item_prefix, grid.x + 1));

  CUDA_CALL(cudaFree(workspace));

  compact_hashmap<SAM_CUDA_BLOCK_SIZE, SAM_CUDA_TILE_SIZE><<<grid, block, 0, stream>>>(
      input, num_input, device_table, item_prefix, unique, num_unique);
  CUDA_CALL(cudaGetLastError());
  CUDA_CALL(cudaFree(item_prefix));
}

void OrderedHashTable::FillWithUnique(const uint32_t *const input,
                                      const size_t num_input,
                                      cudaStream_t stream) {

  const uint32_t num_tiles = (num_input + SAM_CUDA_TILE_SIZE - 1) / SAM_CUDA_TILE_SIZE;

  const dim3 grid(num_tiles);
  const dim3 block(SAM_CUDA_BLOCK_SIZE);

  auto device_table = MutableDeviceOrderedHashTable(this);

  generate_hashmap_unique<SAM_CUDA_BLOCK_SIZE, SAM_CUDA_TILE_SIZE>
      <<<grid, block, 0, stream>>>(input, num_input, device_table);

  CUDA_CALL(cudaGetLastError());
}

} // namespace cuda
} // namespace common
} // namespace samgraph