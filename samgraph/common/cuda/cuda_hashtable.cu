#include <cuda_runtime.h>

#include <cassert>
#include <cstdio>
#include <cub/cub.cuh>

#include "../common.h"
#include "../device.h"
#include "../logging.h"
#include "../timer.h"
#include "cuda_hashtable.h"
#include "cuda_utils.h"

namespace samgraph {
namespace common {
namespace cuda {

class MutableDeviceOrderedHashTable : public DeviceOrderedHashTable {
 public:
  typedef typename DeviceOrderedHashTable::BucketO2N *IteratorO2N;
  typedef typename DeviceOrderedHashTable::BucketN2O *IteratorN2O;

  explicit MutableDeviceOrderedHashTable(OrderedHashTable *const host_table)
      : DeviceOrderedHashTable(host_table->DeviceHandle()) {}

  inline __device__ IteratorO2N SearchO2N(const IdType id) {
    const IdType pos = SearchForPositionO2N(id);

    return GetMutableO2N(pos);
  }

  inline __device__ bool AttemptInsertAtO2N(const IdType pos, const IdType id,
                                            const IdType index,
                                            const IdType version) {
    auto iter = GetMutableO2N(pos);
    const IdType key =
        atomicCAS(&(iter->key), Constant::kEmptyKey, id);
    if (key == Constant::kEmptyKey) {
      iter->index = index;
      iter->version = version;
      return true;
    }
    return key == id;
  }

  inline __device__ IteratorO2N InsertO2N(const IdType id, const IdType index,
                                          const IdType version) {
    IdType pos = HashO2N(id);

    // linearly scan for an empty slot or matching entry
    IdType delta = 1;
    while (!AttemptInsertAtO2N(pos, id, index, version)) {
      pos = HashO2N(pos + delta);
      delta += 1;
    }

    return GetMutableO2N(pos);
  }

  inline __device__ IteratorN2O InsertN2O(const IdType pos,
                                          const IdType global) {
    GetMutableN2O(pos)->global = global;
    return GetMutableN2O(pos);
  }

 private:
  inline __device__ IteratorO2N GetMutableO2N(const IdType pos) {
    assert(pos < this->_o2n_size);
    // The parent class Device is read-only, but we ensure this can only be
    // constructed from a mutable version of OrderedHashTable, making this
    // a safe cast to perform.
    return const_cast<IteratorO2N>(this->_o2n_table + pos);
  }

  inline __device__ IteratorN2O GetMutableN2O(const IdType pos) {
    assert(pos < this->_n2o_size);
    return const_cast<IteratorN2O>(this->_n2o_table + pos);
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

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
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
      table.InsertO2N(items[index], index, version);
    }
  }
}

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void generate_hashmap_unique(const IdType *const items,
                                        const size_t num_items,
                                        MutableDeviceOrderedHashTable table,
                                        const IdType global_offset,
                                        const IdType version) {
  assert(BLOCK_SIZE == blockDim.x);

  using IteratorO2N = typename MutableDeviceOrderedHashTable::IteratorO2N;

  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_items) {
      const IteratorO2N bucket = table.InsertO2N(items[index], index, version);
      IdType pos = global_offset + static_cast<IdType>(index);
      // since we are only inserting unique items, we know their local id
      // will be equal to their index
      bucket->local = pos;
      table.InsertN2O(pos, items[index]);
    }
  }
}

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void count_hashmap(const IdType *items, const size_t num_items,
                              DeviceOrderedHashTable table,
                              IdType *const num_unique, const IdType version) {
  assert(BLOCK_SIZE == blockDim.x);

  using BlockReduce = typename cub::BlockReduce<IdType, BLOCK_SIZE>;
  using BucketO2N = typename DeviceOrderedHashTable::BucketO2N;

  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  IdType count = 0;

#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_items) {
      const BucketO2N &bucket = *table.SearchO2N(items[index]);
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

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void generate_count_hashmap_duplicates(
    const IdType *const items, const size_t num_items,
    MutableDeviceOrderedHashTable table, 
    IdType *const num_unique, const IdType version) {
  assert(BLOCK_SIZE == blockDim.x);

  using BlockReduce = typename cub::BlockReduce<IdType, BLOCK_SIZE>;
  using BucketO2N = typename DeviceOrderedHashTable::BucketO2N;

  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  IdType count = 0;
#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_items) {
      BucketO2N *iter = table.InsertO2N(items[index], index, version);
      if (iter->index == index && iter->version == version) {
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

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void gen_count_hashmap_neighbour(
    const IdType *const items, const size_t num_items,
    const IdType *const indptr, const IdType *const indices,
    MutableDeviceOrderedHashTable table,
    IdType *const num_unique, 
    IdType *const block_max_degree, 
    const IdType version) {
  assert(BLOCK_SIZE == blockDim.x);

  using BlockReduce = typename cub::BlockReduce<IdType, BLOCK_SIZE>;
  using BucketO2N = typename DeviceOrderedHashTable::BucketO2N;

  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  IdType count = 0;
  IdType thread_max_degree = 0;
#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_items) {
      const IdType orig_id = items[index];
      const IdType off = indptr[orig_id];
      const IdType off_end = indptr[orig_id + 1];
      thread_max_degree = (thread_max_degree > (off_end-off)) ? thread_max_degree : (off_end-off) ;
      for (IdType j = off; j < off_end; j++) {
        const IdType nbr_orig_id = indices[j];
        BucketO2N *iter = table.InsertO2N(nbr_orig_id, index, version);
        if (iter->index == index && iter->version == version) {
          ++count;
        }
      }
    }
  }

  __shared__ typename BlockReduce::TempStorage temp_space1;
  __shared__ typename BlockReduce::TempStorage temp_space2;
  IdType max_degree = BlockReduce(temp_space1).Reduce(thread_max_degree, cub::Max());
  count = BlockReduce(temp_space2).Sum(count);

  if (threadIdx.x == 0) {
    num_unique[blockIdx.x] = count;
    block_max_degree[blockIdx.x] = max_degree;
    if (blockIdx.x == 0) {
      num_unique[gridDim.x] = 0;
    }
  }
}

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void gen_count_hashmap_neighbour_single_loop(
    const IdType *const items, const size_t num_items,
    const IdType *const indptr, const IdType *const indices,
    MutableDeviceOrderedHashTable table,
    IdType *const num_unique, 
    IdType *const block_max_iter, 
    const IdType version) {
  assert(BLOCK_SIZE == blockDim.x);

  using BlockReduce = typename cub::BlockReduce<IdType, BLOCK_SIZE>;
  using BucketO2N = typename DeviceOrderedHashTable::BucketO2N;

  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  IdType count = 0;
  __shared__ typename BlockReduce::TempStorage temp_space1;
  __shared__ typename BlockReduce::TempStorage temp_space2;
  IdType iter = 0;
  IdType cur_index = threadIdx.x + block_start;
  IdType cur_j = 0;
  IdType cur_off = 0;
  IdType cur_len = 0;
  const size_t thread_end = (num_items < block_end) ? num_items : block_end;
  if (cur_index < thread_end) {
    auto orig_id = items[cur_index];
    cur_off = indptr[orig_id];
    cur_len = indptr[orig_id + 1] - cur_off;
  }
  while (true) {
    if (cur_index >= thread_end) {
      break;
    } else if (cur_j >= cur_len) {
      cur_index += BLOCK_SIZE;
      if (cur_index < thread_end) {
        auto orig_id = items[cur_index];
        cur_off = indptr[orig_id];
        cur_len = indptr[orig_id + 1] - cur_off;
        cur_j = 0;
      }
    } else {
      const IdType nbr_orig_id = indices[cur_j + cur_off];
      BucketO2N *bucket = table.InsertO2N(nbr_orig_id, cur_index, version);
      count += (bucket->index == cur_index && bucket->version == version);
      cur_j++;
    }
    iter++;
  }
  IdType max_iter = BlockReduce(temp_space1).Reduce(iter, cub::Max());

  count = BlockReduce(temp_space2).Sum(count);

  if (threadIdx.x == 0) {
    num_unique[blockIdx.x] = count;
    block_max_iter[blockIdx.x] = max_iter;
    if (blockIdx.x == 0) {
      num_unique[gridDim.x] = 0;
    }
  }
}

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void compact_hashmap(const IdType *const items,
                                const size_t num_items,
                                MutableDeviceOrderedHashTable table,
                                const IdType *const num_items_prefix,
                                IdType *const num_unique_items,
                                const IdType global_offset,
                                const IdType version) {
  assert(BLOCK_SIZE == blockDim.x);

  using FlagType = IdType;
  using BlockScan = typename cub::BlockScan<FlagType, BLOCK_SIZE>;
  using BucketO2N = typename DeviceOrderedHashTable::BucketO2N;

  constexpr const IdType VALS_PER_THREAD = TILE_SIZE / BLOCK_SIZE;

  __shared__ typename BlockScan::TempStorage temp_space;

  const IdType offset = num_items_prefix[blockIdx.x];

  BlockPrefixCallbackOp<FlagType> prefix_op(0);

  // count successful placements
  for (IdType i = 0; i < VALS_PER_THREAD; ++i) {
    const IdType index = threadIdx.x + i * BLOCK_SIZE + blockIdx.x * TILE_SIZE;

    FlagType flag;
    BucketO2N *kv;
    if (index < num_items) {
      kv = table.SearchO2N(items[index]);
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
      table.InsertN2O(pos, items[index]);
    }
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *num_unique_items = global_offset + num_items_prefix[gridDim.x];
  }
}

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void compact_hashmap_revised(
                                const IdType *const items,
                                const size_t num_items,
                                MutableDeviceOrderedHashTable table,
                                const IdType *const num_items_prefix,
                                const IdType global_offset,
                                const IdType version) {
  assert(BLOCK_SIZE == blockDim.x);

  using FlagType = IdType;
  using BlockScan = typename cub::BlockScan<FlagType, BLOCK_SIZE>;
  using BucketO2N = typename DeviceOrderedHashTable::BucketO2N;

  constexpr const IdType VALS_PER_THREAD = TILE_SIZE / BLOCK_SIZE;

  __shared__ typename BlockScan::TempStorage temp_space;

  const IdType offset = num_items_prefix[blockIdx.x];

  BlockPrefixCallbackOp<FlagType> prefix_op(0);

  // count successful placements
  for (IdType i = 0; i < VALS_PER_THREAD; ++i) {
    const IdType index = threadIdx.x + i * BLOCK_SIZE + blockIdx.x * TILE_SIZE;

    FlagType flag;
    BucketO2N *kv;
    if (index < num_items) {
      kv = table.SearchO2N(items[index]);
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
      table.InsertN2O(pos, items[index]);
    }
  }

  // if (threadIdx.x == 0 && blockIdx.x == 0) {
  //   *num_unique_items = global_offset + num_items_prefix[gridDim.x];
  // }
}

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void compact_hashmap_neighbour(
      const IdType *const items, const size_t num_items,
      const IdType *const indptr, const IdType *const indices,
      MutableDeviceOrderedHashTable table,
      const IdType *const num_items_prefix,
      const IdType *const block_max_degree,
      IdType *const num_unique_items,
      const IdType global_offset,
      const IdType version) {
  assert(BLOCK_SIZE == blockDim.x);

  using FlagType = IdType;
  using BlockScan = typename cub::BlockScan<FlagType, BLOCK_SIZE>;
  using BucketO2N = typename DeviceOrderedHashTable::BucketO2N;

  constexpr const IdType VALS_PER_THREAD = TILE_SIZE / BLOCK_SIZE;

  __shared__ typename BlockScan::TempStorage temp_space;

  const IdType offset = num_items_prefix[blockIdx.x];

  BlockPrefixCallbackOp<FlagType> prefix_op(0);

  // count successful placements
  for (IdType i = 0; i < VALS_PER_THREAD; ++i) {
    const IdType index = threadIdx.x + i * BLOCK_SIZE + blockIdx.x * TILE_SIZE;
    
    FlagType flag;
    IdType orig_id;
    IdType off;
    IdType len = 0;
    if (index < num_items) {
      orig_id = items[index];
      off = indptr[orig_id];
      len = indptr[orig_id + 1] - off;
    }
    assert(block_max_degree[blockIdx.x] >= len);

    for (IdType j = 0; j < block_max_degree[blockIdx.x]; j++) {
      BucketO2N *kv;
      if (j < len) {
        const IdType nbr_orig_id = indices[off + j];
        kv = table.SearchO2N(nbr_orig_id);
        flag = kv->version == version && kv->index == index;
      } else {
        flag = 0;
      }
      if (!flag) kv = nullptr;
      
      BlockScan(temp_space).ExclusiveSum(flag, flag, prefix_op);
      __syncthreads();
      
      if (kv) {
        const IdType pos = global_offset + offset + flag;
        kv->local = pos;
        table.InsertN2O(pos, items[index]);
      }
    }
  }
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *num_unique_items = global_offset + num_items_prefix[gridDim.x];
  }
}

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void compact_hashmap_neighbour_single_loop(
      const IdType *const items, const size_t num_items,
      const IdType *const indptr, const IdType *const indices,
      MutableDeviceOrderedHashTable table,
      const IdType *const num_items_prefix,
      const IdType *const block_max_iter,
      IdType *const num_unique_items,
      const IdType global_offset,
      const IdType version) {
  assert(BLOCK_SIZE == blockDim.x);

  using FlagType = IdType;
  using BlockScan = typename cub::BlockScan<FlagType, BLOCK_SIZE>;
  using BucketO2N = typename DeviceOrderedHashTable::BucketO2N;

  constexpr const IdType VALS_PER_THREAD = TILE_SIZE / BLOCK_SIZE;

  __shared__ typename BlockScan::TempStorage temp_space;

  const IdType offset = num_items_prefix[blockIdx.x];

  BlockPrefixCallbackOp<FlagType> prefix_op(0);

  IdType cur_index = threadIdx.x + TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);
  IdType cur_j = 0;
  IdType cur_off = 0;
  IdType cur_len = 0;
  const size_t thread_end = (num_items < block_end) ? num_items : block_end;
  if (cur_index < thread_end) {
    auto orig_id = items[cur_index];
    cur_off = indptr[orig_id];
    cur_len = indptr[orig_id + 1] - cur_off;
  }
  for (IdType iter = 0; iter < block_max_iter[blockIdx.x]; iter++) {
    FlagType flag = 0;
    BucketO2N *kv = nullptr;
    IdType nbr_orig_id;
    if (cur_index >= thread_end) {
    } else if (cur_j >= cur_len) {
      cur_index += BLOCK_SIZE;
      if (cur_index < thread_end) {
        auto orig_id = items[cur_index];
        cur_off = indptr[orig_id];
        cur_len = indptr[orig_id + 1] - cur_off;
        cur_j = 0;
      }
    } else {
      nbr_orig_id = indices[cur_j + cur_off];
      kv = table.SearchO2N(nbr_orig_id);
      flag = (kv->index == cur_index && kv->version == version);
      if (!flag) kv = nullptr;
      cur_j++;
    }

    BlockScan(temp_space).ExclusiveSum(flag, flag, prefix_op);
    __syncthreads();
    
    if (kv) {
      const IdType pos = global_offset + offset + flag;
      kv->local = pos;
      table.InsertN2O(pos, nbr_orig_id);
    }
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *num_unique_items = global_offset + num_items_prefix[gridDim.x];
  }
}

// DeviceOrderedHashTable implementation
DeviceOrderedHashTable::DeviceOrderedHashTable(const BucketO2N *const o2n_table,
                                               const BucketN2O *const n2o_table,
                                               const size_t o2n_size,
                                               const size_t n2o_size)
    : _o2n_table(o2n_table),
      _n2o_table(n2o_table),
      _o2n_size(o2n_size),
      _n2o_size(n2o_size) {}

DeviceOrderedHashTable OrderedHashTable::DeviceHandle() const {
  return DeviceOrderedHashTable(_o2n_table, _n2o_table, _o2n_size, _n2o_size);
}

// OrderedHashTable implementation
OrderedHashTable::OrderedHashTable(const size_t size, Context ctx,
                                   const size_t scale)
    : _o2n_table(nullptr),
      _o2n_size(TableSize(size, scale)),
      _n2o_size(size),
      _ctx(ctx),
      _version(0),
      _num_items(0) {
  // make sure we will at least as many buckets as items.
  auto device = Device::Get(_ctx);

  _o2n_table = static_cast<BucketO2N *>(
      device->AllocDataSpace(_ctx, sizeof(BucketO2N) * _o2n_size));
  _n2o_table = static_cast<BucketN2O *>(
      device->AllocDataSpace(_ctx, sizeof(BucketN2O) * _n2o_size));

  CUDA_CALL(cudaMemset(_o2n_table, (int)Constant::kEmptyKey,
                       sizeof(BucketO2N) * _o2n_size));
  CUDA_CALL(cudaMemset(_n2o_table, (int)Constant::kEmptyKey,
                       sizeof(BucketN2O) * _n2o_size));
  LOG(INFO) << "cuda hashtable init with " << _o2n_size
            << " O2N table size and " << _n2o_size << " N2O table size";
}

OrderedHashTable::~OrderedHashTable() {
  Timer t;

  auto device = Device::Get(_ctx);
  device->FreeDataSpace(_ctx, _o2n_table);
  device->FreeDataSpace(_ctx, _n2o_table);

  LOG(DEBUG) << "free " << t.Passed();
}

void OrderedHashTable::Reset(StreamHandle stream) {
  auto cu_stream = static_cast<cudaStream_t>(stream);
  CUDA_CALL(cudaMemsetAsync(_o2n_table, (int)Constant::kEmptyKey,
                            sizeof(BucketO2N) * _o2n_size, cu_stream));
  CUDA_CALL(cudaMemsetAsync(_n2o_table, (int)Constant::kEmptyKey,
                            sizeof(BucketN2O) * _n2o_size, cu_stream));
  Device::Get(_ctx)->StreamSync(_ctx, stream);
  _version = 0;
  _num_items = 0;
}

void OrderedHashTable::FillWithDuplicates(const IdType *const input,
                                          const size_t num_input,
                                          IdType *const unique,
                                          IdType *const num_unique,
                                          StreamHandle stream) {
  const size_t num_tiles = RoundUpDiv(num_input, Constant::kCudaTileSize);
  const dim3 grid(num_tiles);
  const dim3 block(Constant::kCudaBlockSize);

  auto device_table = MutableDeviceOrderedHashTable(this);
  auto device = Device::Get(_ctx);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  generate_hashmap_duplicates<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(input, num_input, device_table, _version);
  device->StreamSync(_ctx, stream);

  LOG(DEBUG) << "OrderedHashTable::FillWithDuplicates "
                "generate_hashmap_duplicates with "
             << num_input << " inputs";

  IdType *item_prefix = static_cast<IdType *>(
      device->AllocWorkspace(_ctx, sizeof(IdType) * (grid.x + 1)));
  LOG(DEBUG) << "OrderedHashTable::FillWithDuplicates cuda item_prefix malloc "
             << ToReadableSize(sizeof(IdType) * (grid.x + 1));

  count_hashmap<Constant::kCudaBlockSize, Constant::kCudaTileSize>
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

  IdType *gpu_num_unique =
      static_cast<IdType *>(device->AllocWorkspace(_ctx, sizeof(IdType)));
  LOG(DEBUG)
      << "OrderedHashTable::FillWithDuplicates cuda gpu_num_unique malloc "
      << ToReadableSize(sizeof(IdType));

  compact_hashmap<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(input, num_input, device_table,
                                      item_prefix, gpu_num_unique, _num_items,
                                      _version);
  device->StreamSync(_ctx, stream);

  device->CopyDataFromTo(gpu_num_unique, 0, num_unique, 0, sizeof(IdType), _ctx,
                         CPU(), stream);
  device->StreamSync(_ctx, stream);

  // If the number of input equals to 0, the kernel won't
  // be executed then the value of num_unique will be wrong.
  // We have to manually set the num_unique on this situation.
  if (num_input == 0) {
    *num_unique = _num_items;
  }

  LOG(DEBUG) << "OrderedHashTable::FillWithDuplicates num_unique "
             << *num_unique;

  device->CopyDataFromTo(_n2o_table, 0, unique, 0,
                         sizeof(IdType) * (*num_unique), _ctx, _ctx, stream);
  device->StreamSync(_ctx, stream);

  device->FreeWorkspace(_ctx, gpu_num_unique);
  device->FreeWorkspace(_ctx, item_prefix);
  device->FreeWorkspace(_ctx, workspace);

  _version++;
  _num_items = *num_unique;
}

void OrderedHashTable::CopyUnique(IdType *const unique, StreamHandle stream) {
  auto device = Device::Get(_ctx);
  device->CopyDataFromTo(_n2o_table, 0, unique, 0,
    sizeof(IdType) * _num_items, _ctx, _ctx, stream);
}
void OrderedHashTable::RefUnique(const IdType *&unique, IdType * const num_unique) {
  unique = reinterpret_cast<const IdType*>(_n2o_table);
  *num_unique = _num_items;
}

void OrderedHashTable::FillWithDupRevised(const IdType *const input,
                                          const size_t num_input,
                                          StreamHandle stream) {
  if (num_input == 0) return;
  const size_t num_tiles = RoundUpDiv(num_input, Constant::kCudaTileSize);
  const dim3 grid(num_tiles);
  const dim3 block(Constant::kCudaBlockSize);

  auto device_table = MutableDeviceOrderedHashTable(this);
  auto device = Device::Get(_ctx);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  IdType *item_prefix = static_cast<IdType *>(
      device->AllocWorkspace(_ctx, sizeof(IdType) * (grid.x + 1)));
  LOG(DEBUG) << "OrderedHashTable::FillWithDupRevised cuda item_prefix malloc "
             << ToReadableSize(sizeof(IdType) * (grid.x + 1));
  
  // 1. insert into o2n table, collect each block's new insertion count
  generate_count_hashmap_duplicates<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(input, num_input, device_table, item_prefix, _version);
  device->StreamSync(_ctx, stream);

  LOG(DEBUG) << "OrderedHashTable::FillWithDupRevised "
                "generate_count_hashmap_duplicates with "
             << num_input << " inputs";

  // 2. partial sum
  size_t workspace_bytes;
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      nullptr, workspace_bytes, static_cast<IdType *>(nullptr),
      static_cast<IdType *>(nullptr), grid.x + 1, cu_stream));
  device->StreamSync(_ctx, stream);

  void *workspace = device->AllocWorkspace(_ctx, workspace_bytes);
  LOG(DEBUG) << "OrderedHashTable::FillWithDupRevised cuda workspace malloc "
             << ToReadableSize(workspace_bytes);

  CUDA_CALL(cub::DeviceScan::ExclusiveSum(workspace, workspace_bytes,
                                          item_prefix, item_prefix, grid.x + 1,
                                          cu_stream));
  device->StreamSync(_ctx, stream);

  // 3. now each block knows where in n2o to put the node
  compact_hashmap_revised<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(input, num_input, device_table,
                                      item_prefix,
                                      _num_items,
                                      _version);
  device->StreamSync(_ctx, stream);
  IdType tmp;
  device->CopyDataFromTo(item_prefix+grid.x, 0, &tmp, 0, sizeof(IdType), _ctx,
                         CPU(), stream);
  device->StreamSync(_ctx, stream);
  _num_items += tmp;

  LOG(DEBUG) << "OrderedHashTable::FillWithDupRevised num_unique "
             << _num_items;

  device->FreeWorkspace(_ctx, item_prefix);
  device->FreeWorkspace(_ctx, workspace);

  _version++;
}

void OrderedHashTable::FillNeighbours(
    const IdType *const indptr, const IdType *const indices,
    StreamHandle stream) {
  const size_t num_input = _num_items;
  const IdType *const input = reinterpret_cast<IdType*>(_n2o_table);
  
  const size_t num_tiles = RoundUpDiv(num_input, Constant::kCudaTileSize);
  const dim3 grid(num_tiles);
  const dim3 block(Constant::kCudaBlockSize);

  auto device_table = MutableDeviceOrderedHashTable(this);
  auto device = Device::Get(_ctx);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  IdType *item_prefix = static_cast<IdType *>(
      device->AllocWorkspace(_ctx, sizeof(IdType) * (grid.x + 1) * 2));
  IdType *each_block_max_cnt = item_prefix + grid.x + 1;
  LOG(DEBUG) << "OrderedHashTable::FillNeighbours cuda item_prefix malloc "
             << ToReadableSize(sizeof(IdType) * (grid.x + 1));

  // 1. insert into o2n table, collect each block's new insertion count
  gen_count_hashmap_neighbour_single_loop<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(
          input, num_input, indptr, indices, device_table, 
          item_prefix, each_block_max_cnt, _version);
  device->StreamSync(_ctx, stream);

  // 2. partial sum
  size_t workspace_bytes;
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      nullptr, workspace_bytes, static_cast<IdType *>(nullptr),
      static_cast<IdType *>(nullptr), grid.x + 1, cu_stream));
  device->StreamSync(_ctx, stream);

  void *workspace = device->AllocWorkspace(_ctx, workspace_bytes);
  LOG(DEBUG) << "OrderedHashTable::FillNeighbours cuda item_prefix malloc "
             << ToReadableSize(workspace_bytes);

  CUDA_CALL(cub::DeviceScan::ExclusiveSum(workspace, workspace_bytes,
                                          item_prefix, item_prefix, grid.x + 1,
                                          cu_stream));
  device->StreamSync(_ctx, stream);

  IdType *gpu_num_unique =
      static_cast<IdType *>(device->AllocWorkspace(_ctx, sizeof(IdType)));

  // 3.now each block knows where in n2o to put the node
  compact_hashmap_neighbour_single_loop<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(
          input, num_input, indptr, indices, device_table, 
          item_prefix, each_block_max_cnt, 
          gpu_num_unique, _num_items, _version);
  device->StreamSync(_ctx, stream);
  if (num_input != 0) {
    device->CopyDataFromTo(gpu_num_unique, 0, &_num_items, 0, sizeof(IdType), _ctx,
                           CPU(), stream);
    device->StreamSync(_ctx, stream);
  }

  LOG(DEBUG) << "OrderedHashTable::FillWithDuplicates num_unique "
             << _num_items;

  device->FreeWorkspace(_ctx, gpu_num_unique);
  device->FreeWorkspace(_ctx, item_prefix);
  device->FreeWorkspace(_ctx, workspace);

  _version++;
}

void OrderedHashTable::FillWithUnique(const IdType *const input,
                                      const size_t num_input,
                                      StreamHandle stream) {
  const size_t num_tiles = RoundUpDiv(num_input, Constant::kCudaTileSize);
  const dim3 grid(num_tiles);
  const dim3 block(Constant::kCudaBlockSize);

  auto device_table = MutableDeviceOrderedHashTable(this);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  generate_hashmap_unique<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(input, num_input, device_table,
                                      _num_items, _version);
  Device::Get(_ctx)->StreamSync(_ctx, stream);

  _version++;
  _num_items += num_input;

  LOG(DEBUG) << "OrderedHashTable::FillWithUnique insert " << num_input
             << " items, now " << _num_items << " in total";
}

}  // namespace cuda
}  // namespace common
}  // namespace samgraph