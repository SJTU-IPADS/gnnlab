#include <cuda_runtime.h>

#include <cub/cub.cuh>

#include "../constant.h"
#include "../device.h"
#include "../logging.h"
#include "../profiler.h"
#include "../timer.h"
#include "cuda_frequency_hashmap.h"
#include "cuda_utils.h"

namespace samgraph {
namespace common {
namespace cuda {

namespace {

size_t TableSize(const size_t num, const size_t scale) {
  /** SXN: why bother? 1+[log2([n/2])] is similar to [log2(n)]*/
  const size_t next_pow2 = 1 << static_cast<size_t>(1 + std::log2(num >> 1));
  return next_pow2 << scale;
}

class MutableDeviceFrequencyHashmap : public DeviceFrequencyHashmap {
 public:
  typedef typename DeviceFrequencyHashmap::NodeBucket *NodeIterator;
  typedef typename DeviceFrequencyHashmap::EdgeBucket *EdgeIterator;

  explicit MutableDeviceFrequencyHashmap(FrequencyHashmap *const host_map)
      : DeviceFrequencyHashmap(host_map->DeviceHandle()) {}

  inline __device__ NodeIterator SearchNode(const IdType id) {
    const IdType pos = SearchNodeForPosition(id);
    return GetMutableNode(pos);
  }

  inline __device__ EdgeIterator SearchEdge(const IdType node_idx,
                                            const IdType dst) {
    const IdType pos = SearchEdgeForPosition(node_idx, dst);
    return GetMutableEdge(pos);
  }

  inline __device__ bool AttemptInsertNodeAt(const IdType pos,
                                             const IdType id) {
    const IdType key =
        atomicCAS(&GetMutableNode(pos)->key, Constant::kEmptyKey, id);
    if (key == Constant::kEmptyKey || key == id) {
      // printf("Insert %u insert at pos %u\n", id, pos);
      return true;
    } else {
      return false;
    }
  }

  inline __device__ NodeIterator InsertNode(const IdType id) {
    IdType pos = NodeHash(id);

    IdType delta = 1;
    while (!AttemptInsertNodeAt(pos, id)) {
      // printf("node %u insert pos %u fail\n", id, pos);
      pos = NodeHash(pos + delta);
      delta += 1;
    }
    return GetMutableNode(pos);
  }

  inline __device__ bool AttemptInsertEdgeAt(const IdType pos, const IdType src,
                                             const IdType dst,
                                             const IdType index) {
    EdgeIterator edge_iter = GetMutableEdge(pos);
    const IdType key = atomicCAS(&edge_iter->key, Constant::kEmptyKey, dst);
    if (key == Constant::kEmptyKey || key == dst) {
      atomicAdd(&edge_iter->count, 1U);
#ifndef SXN_REVISED
      atomicCAS(&edge_iter->index, Constant::kEmptyKey, index);
      if (key == Constant::kEmptyKey) {
        NodeIterator node_iter = SearchNode(src);
        atomicAdd(&node_iter->count, 1U);
      }
#else
      /** SXN: remove atomic by checking swapped out key */
      if (key == Constant::kEmptyKey) {
        edge_iter->index = index;
        NodeIterator node_iter = SearchNode(src);
        atomicAdd(&node_iter->count, 1U);
      }
#endif
      return true;
    } else {
      return false;
    }
  }

  inline __device__ EdgeIterator InsertEdge(const IdType node_idx,
                                            const IdType src, const IdType dst,
                                            const IdType index) {
    IdType start_off = node_idx * _per_node_etable_size;
    IdType pos = EdgeHash(dst);

    IdType delta = 1;
    while (!AttemptInsertEdgeAt(start_off + pos, src, dst, index)) {
      pos = EdgeHash(pos + delta);
      delta += 1;
    }

    return GetMutableEdge(start_off + pos);
  }

  inline __device__ NodeIterator GetMutableNode(const IdType pos) {
    assert(pos < _ntable_size);
    return const_cast<NodeIterator>(_node_table + pos);
  }

  inline __device__ EdgeIterator GetMutableEdge(const IdType pos) {
    assert(pos < _etable_size);
    return const_cast<EdgeIterator>(_edge_table + pos);
  }

  inline __device__ IdType GetRelativePos(const EdgeIterator iter) {
    return iter - _edge_table;
  }
};

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void init_node_table(MutableDeviceFrequencyHashmap table,
                                const size_t num_bucket) {
  assert(BLOCK_SIZE == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  using NodeIterator = typename MutableDeviceFrequencyHashmap::NodeIterator;

#pragma unroll
  for (IdType index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_bucket) {
      NodeIterator node_iter = table.GetMutableNode(index);
      node_iter->key = Constant::kEmptyKey;
      node_iter->count = 0;
    }
  }
}

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void init_edge_table(MutableDeviceFrequencyHashmap table,
                                const size_t num_bucket) {
  assert(BLOCK_SIZE == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  using EdgeIterator = typename MutableDeviceFrequencyHashmap::EdgeIterator;

#pragma unroll
  for (IdType index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_bucket) {
      EdgeIterator edge_iter = table.GetMutableEdge(index);
      edge_iter->key = Constant::kEmptyKey;
      edge_iter->count = 0;
      edge_iter->index = Constant::kEmptyKey;
    }
  }
}

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void init_unique_range(IdType *_unique_range,
                                  const size_t unique_list_size) {
  assert(BLOCK_SIZE == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  using EdgeIterator = typename MutableDeviceFrequencyHashmap::EdgeIterator;

#pragma unroll
  for (IdType index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < unique_list_size) {
      _unique_range[index] = index;
    }
  }
}

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void reset_node_table(MutableDeviceFrequencyHashmap table,
                                 const IdType *nodes, const size_t num_nodes) {
  assert(BLOCK_SIZE == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  using NodeIterator = typename MutableDeviceFrequencyHashmap::NodeIterator;

#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_nodes) {
      IdType id = nodes[index];
      NodeIterator node_iter = table.SearchNode(id);
      node_iter->key = Constant::kEmptyKey;
      node_iter->count = 0;
    }
  }
}

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void reset_edge_table(MutableDeviceFrequencyHashmap table,
                                 IdType *unique_node_idx, IdType *unique_dst,
                                 const size_t num_unique) {
  assert(BLOCK_SIZE == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  using EdgeIterator = typename MutableDeviceFrequencyHashmap::EdgeIterator;

#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_unique) {
      IdType node_idx = unique_node_idx[index];
      IdType dst = unique_dst[index];
      EdgeIterator edge_iter = table.SearchEdge(node_idx, dst);
      edge_iter->key = Constant::kEmptyKey;
      edge_iter->count = 0;
      edge_iter->index = Constant::kEmptyKey;
    }
  }
}

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void reset_edge_table_revised(MutableDeviceFrequencyHashmap table,
                                 IdType *unique_pos, IdType *unique_dst,
                                 const size_t num_unique) {
  assert(BLOCK_SIZE == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  using EdgeIterator = typename MutableDeviceFrequencyHashmap::EdgeIterator;

#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_unique) {
      IdType dst = unique_dst[index];
      EdgeIterator edge_iter = table.GetMutableEdge(unique_pos[index]);
      assert(edge_iter->key == dst);
      edge_iter->key = Constant::kEmptyKey;
      edge_iter->count = 0;
      edge_iter->index = Constant::kEmptyKey;
    }
  }
}

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void populate_node_table(const IdType *nodes,
                                    const size_t num_input_node,
                                    MutableDeviceFrequencyHashmap table) {
  assert(BLOCK_SIZE == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_input_node) {
      table.InsertNode(nodes[index]);
    }
  }
}

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void count_frequency(const IdType *input_src,
                                const IdType *input_dst,
                                const size_t num_input_edge,
                                const size_t edges_per_node,
                                IdType *item_prefix,
                                MutableDeviceFrequencyHashmap table) {
  assert(BLOCK_SIZE == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  using BlockReduce = typename cub::BlockReduce<IdType, BLOCK_SIZE>;
  using EdgeIterator = typename MutableDeviceFrequencyHashmap::EdgeIterator;

  IdType count = 0;
#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_input_edge && input_src[index] != Constant::kEmptyKey) {
      IdType node_idx = index / edges_per_node;
      EdgeIterator edge_iter =
          table.InsertEdge(node_idx, input_src[index], input_dst[index], index);
      if (edge_iter->index == index) {
        ++count;
      }
    }
  }

  __shared__ typename BlockReduce::TempStorage temp_space;

  count = BlockReduce(temp_space).Sum(count);

  if (threadIdx.x == 0) {
    item_prefix[blockIdx.x] = count;
    if (blockIdx.x == 0) {
      item_prefix[gridDim.x] = 0;
    }
  }
}
template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void count_frequency_revised(
                                IdType *input_src,
                                IdType *input_dst,
                                const size_t num_input_edge,
                                const size_t edges_per_node,
                                IdType *item_prefix,
                                MutableDeviceFrequencyHashmap table) {
  assert(BLOCK_SIZE == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  using BlockReduce = typename cub::BlockReduce<IdType, BLOCK_SIZE>;
  using EdgeIterator = typename MutableDeviceFrequencyHashmap::EdgeIterator;

  IdType count = 0;
#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_input_edge && input_src[index] != Constant::kEmptyKey) {
      IdType node_idx = index / edges_per_node;
      EdgeIterator edge_iter =
          table.InsertEdge(node_idx, input_src[index], input_dst[index], index);
      input_src[index] = Constant::kEmptyKey;
      if (edge_iter->index == index) {
        input_src[index] = table.GetRelativePos(edge_iter);
        ++count;
      }
    }
  }

  __shared__ typename BlockReduce::TempStorage temp_space;

  count = BlockReduce(temp_space).Sum(count);

  if (threadIdx.x == 0) {
    item_prefix[blockIdx.x] = count;
    if (blockIdx.x == 0) {
      item_prefix[gridDim.x] = 0;
    }
  }
}

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void generate_unique_edges(
    const IdType *input_src, const IdType *input_dst,
    const size_t num_input_edge, IdType *item_prefix, IdType *unique_node_idx,
    IdType *unique_src, IdType *unique_dst, IdType *unique_count,
    size_t *num_unique, const size_t edges_per_node,
    MutableDeviceFrequencyHashmap table) {
  assert(BLOCK_SIZE == blockDim.x);

  using FlagType = IdType;
  using BlockScan = typename cub::BlockScan<FlagType, BLOCK_SIZE>;
  using EdgeBucket = typename DeviceFrequencyHashmap::EdgeBucket;

  constexpr const IdType VALS_PER_THREAD = TILE_SIZE / BLOCK_SIZE;

  __shared__ typename BlockScan::TempStorage temp_space;

  const IdType offset = item_prefix[blockIdx.x];

  BlockPrefixCallbackOp<FlagType> prefix_op(0);

  // count successful placements
  for (IdType i = 0; i < VALS_PER_THREAD; ++i) {
    const IdType index = threadIdx.x + i * BLOCK_SIZE + blockIdx.x * TILE_SIZE;

    IdType node_idx = index / edges_per_node;
    FlagType flag;
    EdgeBucket *bucket;
    if (index < num_input_edge && input_src[index] != Constant::kEmptyKey) {
      bucket = table.SearchEdge(node_idx, input_dst[index]);
      flag = bucket->index == index;
    } else {
      flag = 0;
    }

    if (!flag) {
      bucket = nullptr;
    }

    BlockScan(temp_space).ExclusiveSum(flag, flag, prefix_op);
    __syncthreads();

    if (bucket) {
      const IdType pos = offset + flag;
      unique_node_idx[pos] = node_idx;
      unique_src[pos] = input_src[index];
      unique_dst[pos] = input_dst[index];
      unique_count[pos] = bucket->count;
    }
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *num_unique = item_prefix[gridDim.x];
  }
}

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void generate_unique_edges_pos(
    IdType *input_src,
    const IdType *input_nodes,
    const size_t num_input_node,
    const size_t num_input_edge, IdType *item_prefix,
    Id64Type * unique_combination_key,
    IdType *unique_edge_pos, const size_t edges_per_node,
    MutableDeviceFrequencyHashmap table) {
  assert(BLOCK_SIZE == blockDim.x);

  using FlagType = IdType;
  using BlockScan = typename cub::BlockScan<FlagType, BLOCK_SIZE>;
  using EdgeBucket = typename DeviceFrequencyHashmap::EdgeBucket;

  constexpr const IdType VALS_PER_THREAD = TILE_SIZE / BLOCK_SIZE;

  __shared__ typename BlockScan::TempStorage temp_space;

  const IdType offset = item_prefix[blockIdx.x];

  BlockPrefixCallbackOp<FlagType> prefix_op(0);

  // count successful placements
  for (IdType i = 0; i < VALS_PER_THREAD; ++i) {
    const IdType index = threadIdx.x + i * BLOCK_SIZE + blockIdx.x * TILE_SIZE;

    IdType node_idx = index / edges_per_node;
    FlagType flag = 0;
    EdgeBucket *bucket = nullptr;
    if (index < num_input_edge && input_src[index] != Constant::kEmptyKey) {
      /** SXN: optimize: input dst can be modified to location in hash_table,
       * thus no need to search */
      bucket = table.GetMutableEdge(input_src[index]);
      flag = 1;
    }

    BlockScan(temp_space).ExclusiveSum(flag, flag, prefix_op);
    __syncthreads();

    if (bucket) {
      const IdType pos = offset + flag;
      unique_edge_pos[pos] = table.GetRelativePos(bucket);
      unique_combination_key[pos] = 
          (Id64Type)((num_input_node - node_idx) << 32) | 
          ((Id64Type)bucket->count); 
    }
  }
}

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void construct_unique_edge_list(
    const Id64Type *unique_combination_key,
    const IdType *unique_pos,
    IdType *unique_node_idx,
    IdType *unique_dst,
    const size_t num_unique,
    MutableDeviceFrequencyHashmap table) {
  assert(BLOCK_SIZE == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_unique) {
      unique_node_idx[index] = table.PosToNodeIdx(unique_pos[index]);
      unique_dst[index] = table.GetMutableEdge(unique_pos[index])->key;
    }
  }
}

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void reorder_unique(
    const IdType *unique_src, const IdType *unique_idx,
    const IdType *tmp_unique_node_idx, const IdType *tmp_unique_dst,
    const IdType *tmp_unique_frequency, IdType *unique_node_idx,
    IdType *unique_dst, IdType *unique_frequency,
    Id64Type *unique_combination_key, const size_t num_unique) {
  assert(BLOCK_SIZE == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_unique) {
      IdType origin_idx = unique_idx[index];
      unique_node_idx[index] = tmp_unique_node_idx[origin_idx];
      unique_dst[index] = tmp_unique_dst[origin_idx];
      unique_frequency[index] = tmp_unique_frequency[origin_idx];
      unique_combination_key[index] =
          (((Id64Type)unique_src[index]) << 32) |
          ((Id64Type)tmp_unique_frequency[origin_idx]);
    }
  }
}

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void generate_num_edge(const IdType *nodes, const size_t num_nodes,
                                  const size_t K, IdType *num_edge_prefix,
                                  IdType *num_output_prefix,
                                  DeviceFrequencyHashmap table) {
  assert(BLOCK_SIZE == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  using NodeBucket = typename DeviceFrequencyHashmap::NodeBucket;

#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_nodes) {
      const NodeBucket &bucket = *table.SearchNode(nodes[index]);
      num_edge_prefix[index] = bucket.count;
      num_output_prefix[index] = bucket.count > K ? K : bucket.count;
    }
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    num_edge_prefix[num_nodes] = 0;
    num_output_prefix[num_nodes] = 0;
  }
}

__global__ void compact_output(const IdType *unique_src,
                               const IdType *unique_dst,
                               const IdType *unique_frequency,
                               const size_t num_nodes, const size_t K,
                               const IdType *num_unique_prefix,
                               const IdType *num_output_prefix,
                               IdType *output_src, IdType *output_dst,
                               IdType *output_data, size_t *num_output) {
  size_t i = blockIdx.x * blockDim.y + threadIdx.y;
  const size_t stride = blockDim.y * gridDim.x;

  /** SXN: this loop `may` be unnecessary */
  while (i < num_nodes) {
    IdType k = threadIdx.x;
    IdType max_output = num_output_prefix[i + 1] - num_output_prefix[i];
    /** SXN: max_output must <= K, ensured in generate_num_edge */
    while (k < K && k < max_output) {
      IdType from_off = num_unique_prefix[i] + k;
      IdType to_off = num_output_prefix[i] + k;

      output_src[to_off] = unique_src[from_off];
      output_dst[to_off] = unique_dst[from_off];
      output_data[to_off] = unique_frequency[from_off];

      k += blockDim.x;
    }

    i += stride;
  }

  if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
    *num_output = num_output_prefix[num_nodes];
  }
}
__global__ void compact_output_revised(
    const IdType* input_nodes,
    const Id64Type *unique_combination_key,
    const IdType *unique_dst,
    const size_t num_nodes, const size_t K,
    const IdType *num_unique_prefix,
    const IdType *num_output_prefix,
    IdType *output_src, IdType *output_dst,
    IdType *output_data, size_t *num_output) {
  size_t i = blockIdx.x * blockDim.y + threadIdx.y;
  const size_t stride = blockDim.y * gridDim.x;

  /** SXN: this loop `may` be unnecessary */
  if (i < num_nodes) {
    IdType k = threadIdx.x;
    IdType max_output = num_output_prefix[i + 1] - num_output_prefix[i];
    /** SXN: max_output must <= K, ensured in generate_num_edge */
    while (k < max_output) {
      IdType from_off = num_unique_prefix[i] + k;
      IdType to_off = num_output_prefix[i] + k;

      output_src[to_off] = input_nodes[num_nodes - (unique_combination_key[from_off] >> 32)];
      output_data[to_off] = (unique_combination_key[from_off]);
      output_dst[to_off] = unique_dst[from_off];

      k += blockDim.x;
    }
  }

  if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
    *num_output = num_output_prefix[num_nodes];
  }
}

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void validate_combine_w_freq(const IdType *unique_src,
                                        IdType *unique_frequency,
                                        Id64Type *unique_combination_key,
                                        const size_t num_unique) {
  assert(BLOCK_SIZE == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_unique) {
      IdType original_node_id = unique_src[index];
      Id64Type ck = unique_combination_key[index];
      IdType extracted_node_id = (ck >> 32) & 0x00000000ffffffff;
      IdType extracted_freq = ck & 0x00000000ffffffff;
      assert(extracted_node_id == original_node_id);
      assert(extracted_freq == unique_frequency[index]);
    }
  }
}
template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void validate_combine(const IdType *unique_src,
                                 Id64Type *unique_combination_key,
                                 const size_t num_unique) {
  assert(BLOCK_SIZE == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_unique) {
      IdType original_node_id = unique_src[index];
      Id64Type ck = unique_combination_key[index];
      IdType extracted_node_id = (ck >> 32) & 0x00000000ffffffff;
      assert(extracted_node_id == original_node_id);
    }
  }
}

}  // namespace

DeviceFrequencyHashmap::DeviceFrequencyHashmap(
    const NodeBucket *node_table, const EdgeBucket *edge_table,
    const size_t ntable_size, const size_t etable_size,
    const size_t per_node_etable_size)
    : _node_table(node_table),
      _edge_table(edge_table),
      _ntable_size(ntable_size),
      _etable_size(etable_size),
      _per_node_etable_size(per_node_etable_size) {}

DeviceFrequencyHashmap FrequencyHashmap::DeviceHandle() const {
  return DeviceFrequencyHashmap(_node_table, _edge_table, _ntable_size,
                                _etable_size, _per_node_etable_size);
}

FrequencyHashmap::FrequencyHashmap(const size_t max_nodes,
                                   const size_t edges_per_node, Context ctx,
                                   const size_t node_table_scale,
                                   const size_t edge_table_scale)
    : _ctx(ctx),
      _max_nodes(max_nodes),
      _edges_per_node(edges_per_node),
      _ntable_size(TableSize(max_nodes, node_table_scale)),
      _etable_size(max_nodes * TableSize(edges_per_node, edge_table_scale)),
      _per_node_etable_size(TableSize(edges_per_node, edge_table_scale)),
      _num_node(0),
      _node_list_size(max_nodes),
      _num_unique(0),
      _unique_list_size(max_nodes * edges_per_node) {
  auto device = Device::Get(_ctx);
  CHECK_EQ(_ctx.device_type, kGPU);

  _node_table = static_cast<NodeBucket *>(
      device->AllocDataSpace(_ctx, sizeof(NodeBucket) * _ntable_size));
  _edge_table = static_cast<EdgeBucket *>(
      device->AllocDataSpace(_ctx, sizeof(EdgeBucket) * _etable_size));

  _node_list = static_cast<IdType *>(
      device->AllocDataSpace(_ctx, sizeof(IdType) * _node_list_size));

#ifndef SXN_REVISED
  _unique_range = static_cast<IdType *>(
      device->AllocDataSpace(_ctx, sizeof(IdType) * _unique_list_size));
  _unique_node_idx = static_cast<IdType *>(
      device->AllocDataSpace(_ctx, sizeof(IdType) * _unique_list_size));
  _unique_src = static_cast<IdType *>(
      device->AllocDataSpace(_ctx, sizeof(IdType) * _unique_list_size));
  _unique_dst = static_cast<IdType *>(
      device->AllocDataSpace(_ctx, sizeof(IdType) * _unique_list_size));
  _unique_frequency = static_cast<IdType *>(
      device->AllocDataSpace(_ctx, sizeof(IdType) * _unique_list_size));
  _unique_combination_key = static_cast<Id64Type *>(
      device->AllocDataSpace(_ctx, sizeof(Id64Type) * _unique_list_size));
#else
  _unique_node_idx = static_cast<IdType *>(
      device->AllocDataSpace(_ctx, sizeof(IdType) * _unique_list_size * 2));
  _unique_dst = _unique_node_idx + _unique_list_size;
  _unique_combination_key = static_cast<Id64Type *>(
      device->AllocDataSpace(_ctx, sizeof(Id64Type) * _unique_list_size));
#endif

  auto device_table = MutableDeviceFrequencyHashmap(this);
  dim3 grid0(RoundUpDiv(_ntable_size, Constant::kCudaTileSize));
  dim3 grid1(RoundUpDiv(_etable_size, Constant::kCudaTileSize));
  dim3 grid2(RoundUpDiv(_unique_list_size, Constant::kCudaTileSize));
  dim3 block0(Constant::kCudaBlockSize);
  dim3 block1(Constant::kCudaBlockSize);
  dim3 block2(Constant::kCudaBlockSize);

  init_node_table<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid0, block0>>>(device_table, _ntable_size);
  init_edge_table<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid1, block1>>>(device_table, _etable_size);
#ifndef SXN_REVISED
  init_unique_range<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid2, block2>>>(_unique_range, _unique_list_size);
#else
#endif

  LOG(INFO) << "FrequencyHashmap init with node table size: " << _ntable_size
            << " and edge table size: " << _etable_size;
}

FrequencyHashmap::~FrequencyHashmap() {
  auto device = Device::Get(_ctx);

  device->FreeDataSpace(_ctx, _node_table);
  device->FreeDataSpace(_ctx, _edge_table);
  device->FreeDataSpace(_ctx, _node_list);
#ifndef SXN_REVISED
  device->FreeDataSpace(_ctx, _unique_range);
  device->FreeDataSpace(_ctx, _unique_node_idx);
  device->FreeDataSpace(_ctx, _unique_src);
  device->FreeDataSpace(_ctx, _unique_dst);
  device->FreeDataSpace(_ctx, _unique_frequency);
#else
  device->FreeDataSpace(_ctx, _unique_node_idx);
#endif
  device->FreeDataSpace(_ctx, _unique_combination_key);
}

#ifndef SXN_REVISED
void FrequencyHashmap::GetTopK(const IdType *input_src, const IdType *input_dst,
                               const size_t num_input_edge,
                               const IdType *input_nodes,
                               const size_t num_input_node, const size_t K,
                               IdType *output_src, IdType *output_dst,
                               IdType *output_data, size_t *num_output,
                               StreamHandle stream, uint64_t task_key) {
  const size_t num_tiles0 = RoundUpDiv(num_input_node, Constant::kCudaTileSize);
  const size_t num_tiles1 = RoundUpDiv(num_input_edge, Constant::kCudaTileSize);
  const dim3 grid0(num_tiles0);
  const dim3 grid1(num_tiles1);

  const dim3 block0(Constant::kCudaBlockSize);
  const dim3 block1(Constant::kCudaBlockSize);

  dim3 block2(Constant::kCudaBlockSize, 1);
  while (static_cast<size_t>(block2.x) >= 2 * K) {
    block2.x /= 2;
    block2.y *= 2;
  }
  dim3 grid2(RoundUpDiv(num_input_node, static_cast<size_t>(block2.y)));

  auto device_table = MutableDeviceFrequencyHashmap(this);
  auto device = Device::Get(_ctx);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  size_t workspace_bytes0;
  size_t workspace_bytes1;
  size_t workspace_bytes2;
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      nullptr, workspace_bytes0, static_cast<IdType *>(nullptr),
      static_cast<IdType *>(nullptr), grid0.x + 1, cu_stream));
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      nullptr, workspace_bytes1, static_cast<IdType *>(nullptr),
      static_cast<IdType *>(nullptr), grid1.x + 1, cu_stream));
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      nullptr, workspace_bytes2, static_cast<IdType *>(nullptr),
      static_cast<IdType *>(nullptr), num_input_node + 1, cu_stream));
  device->StreamSync(_ctx, stream);

  void *workspace0 = device->AllocWorkspace(_ctx, workspace_bytes0);
  void *workspace1 = device->AllocWorkspace(_ctx, workspace_bytes1);
  void *workspace2 = device->AllocWorkspace(_ctx, workspace_bytes2);

  // 1. populate the node table
  Timer t1;
  populate_node_table<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid0, block0, 0, cu_stream>>>(input_nodes, num_input_node,
                                        device_table);
  device->StreamSync(_ctx, stream);
  double step1_time = t1.Passed();

  LOG(DEBUG) << "FrequencyHashmap::GetTopK step 1 finish with "
             << num_input_node << " input nodes with grid " << grid0.x
             << " block " << block0.x;

  // 2. count frequency of every unique edge and
  //    count unique edges for every node
  Timer t2;
  IdType *num_unique_prefix = static_cast<IdType *>(
      device->AllocWorkspace(_ctx, sizeof(IdType) * (grid1.x + 1)));
  count_frequency<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid1, block1, 0, cu_stream>>>(input_src, input_dst, num_input_edge,
                                        _edges_per_node, num_unique_prefix,
                                        device_table);
  device->StreamSync(_ctx, stream);
  double step2_time = t2.Passed();
  LOG(DEBUG) << "FrequencyHashmap::GetTopK step 2 finish";

  // 3. count the number of unique edges.
  //    prefix sum the the array
  Timer t3;

  CUDA_CALL(cub::DeviceScan::ExclusiveSum(workspace1, workspace_bytes1,
                                          num_unique_prefix, num_unique_prefix,
                                          grid1.x + 1, cu_stream));
  device->StreamSync(_ctx, stream);
  double step3_time = t3.Passed();
  LOG(DEBUG) << "FrequencyHashmap::GetTopK step 3 finish";

  // 4. get the array of all unique edges.
  Timer t4;
  size_t *device_num_unique =
      static_cast<size_t *>(device->AllocWorkspace(_ctx, sizeof(size_t)));
  IdType *tmp_unique_node_idx = static_cast<IdType *>(
      device->AllocWorkspace(_ctx, sizeof(IdType) * _unique_list_size));
  IdType *tmp_unique_dst = static_cast<IdType *>(
      device->AllocWorkspace(_ctx, sizeof(IdType) * _unique_list_size));
  IdType *tmp_unique_frequency = static_cast<IdType *>(
      device->AllocWorkspace(_ctx, sizeof(IdType) * _unique_list_size));
  generate_unique_edges<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid1, block1, 0, cu_stream>>>(input_src, input_dst, num_input_edge,
                                        num_unique_prefix, tmp_unique_node_idx,
                                        _unique_src, tmp_unique_dst,
                                        tmp_unique_frequency, device_num_unique,
                                        _edges_per_node, device_table);
  device->StreamSync(_ctx, stream);

  /** SXN: last item of num_unique_prefix. no need to use another
   * device_num_unique */
  device->CopyDataFromTo(device_num_unique, 0, &_num_unique, 0, sizeof(size_t),
                         _ctx, CPU(), stream);
  device->StreamSync(_ctx, stream);

  LOG(DEBUG) << "FrequencyHashmap::GetTopK step 4 finish with number of unique "
             << _num_unique;
  double step4_time = t4.Passed();

  // 5. pair-sort unique array using src as key.
  Timer t5;

  IdType *unique_idx = static_cast<IdType *>(
      device->AllocWorkspace(_ctx, sizeof(IdType) * _num_unique));
  device->CopyDataFromTo(_unique_range, 0, unique_idx, 0,
                         sizeof(IdType) * _num_unique, _ctx, _ctx, stream);

  size_t workspace_bytes3;
  CUDA_CALL(cub::DeviceRadixSort::SortPairsDescending(
      nullptr, workspace_bytes3, static_cast<IdType *>(nullptr),
      static_cast<IdType *>(nullptr), static_cast<IdType *>(nullptr),
      static_cast<IdType *>(nullptr), _num_unique, 0, sizeof(IdType) * 8,
      cu_stream));
  device->StreamSync(_ctx, stream);

  void *workspace3 = device->AllocWorkspace(_ctx, workspace_bytes3);
  CUDA_CALL(cub::DeviceRadixSort::SortPairsDescending(
      workspace3, workspace_bytes3, _unique_src, _unique_src, unique_idx,
      unique_idx, _num_unique, 0, sizeof(IdType) * 8, cu_stream));
  device->StreamSync(_ctx, stream);

  const size_t num_tiles3 = RoundUpDiv(_num_unique, Constant::kCudaTileSize);
  const dim3 grid3(num_tiles3);
  const dim3 block3(Constant::kCudaBlockSize);

  reorder_unique<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid3, block3, 0, cu_stream>>>(
          _unique_src, unique_idx, tmp_unique_node_idx, tmp_unique_dst,
          tmp_unique_frequency, _unique_node_idx, _unique_dst,
          _unique_frequency, _unique_combination_key, _num_unique);
  device->StreamSync(_ctx, stream);
  double step5_time = t5.Passed();
  LOG(DEBUG) << "FrequencyHashmap::GetTopK step 5 finish";

  // 6. sort the unique src node array.
  Timer t6;
  device->CopyDataFromTo(input_nodes, 0, _node_list, 0,
                         num_input_node * sizeof(IdType), _ctx, _ctx, stream);
  device->StreamSync(_ctx, stream);
  _num_node = num_input_node;

  size_t workspace_bytes4;
  CUDA_CALL(cub::DeviceRadixSort::SortKeysDescending(
      nullptr, workspace_bytes4, static_cast<IdType *>(nullptr),
      static_cast<IdType *>(nullptr), num_input_node, 0, sizeof(IdType) * 8,
      cu_stream));
  device->StreamSync(_ctx, stream);

  void *workspace4 = device->AllocWorkspace(_ctx, workspace_bytes4);
  CUDA_CALL(cub::DeviceRadixSort::SortKeysDescending(
      workspace4, workspace_bytes4, _node_list, _node_list, num_input_node, 0,
      sizeof(IdType) * 8, cu_stream));
  device->StreamSync(_ctx, stream);

  double step6_time = t6.Passed();
  LOG(DEBUG) << "FrequencyHashmap::GetTopK step 6 finish";

  // 7. get array unique edge number in the order of src nodes.
  //    also count the number of output edges for each nodes.
  //    prefix sum for array of unique edge number.
  Timer t7;
  IdType *num_edge_prefix = static_cast<IdType *>(
      device->AllocWorkspace(_ctx, (num_input_node + 1) * sizeof(IdType)));
  IdType *num_output_prefix = static_cast<IdType *>(
      device->AllocWorkspace(_ctx, (num_input_node + 1) * sizeof(IdType)));
  generate_num_edge<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid0, block0, 0, cu_stream>>>(_node_list, num_input_node, K,
                                        num_edge_prefix, num_output_prefix,
                                        device_table);
  device->StreamSync(_ctx, stream);

  CUDA_CALL(cub::DeviceScan::ExclusiveSum(workspace2, workspace_bytes2,
                                          num_edge_prefix, num_edge_prefix,
                                          num_input_node + 1, cu_stream));
  device->StreamSync(_ctx, stream);
  double step7_time = t7.Passed();
  LOG(DEBUG) << "FrequencyHashmap::GetTopK step 7 finish";

  // 8. segment-sort the edge for every node using the frequency as key
  //    and the dst as value.
  Timer t8;
  // size_t workspace_bytes5;
  // CUDA_CALL(cub::DeviceSegmentedRadixSort::SortPairsDescending(
  //     nullptr, workspace_bytes5, static_cast<IdType *>(nullptr),
  //     static_cast<IdType *>(nullptr), static_cast<IdType *>(nullptr),
  //     static_cast<IdType *>(nullptr), _num_unique, num_input_node,
  //     static_cast<IdType *>(nullptr), static_cast<IdType *>(nullptr), 0,
  //     sizeof(IdType) * 8, cu_stream));
  // device->StreamSync(_ctx, stream);

  // void *workspace5 = device->AllocWorkspace(_ctx, workspace_bytes5);
  // CUDA_CALL(cub::DeviceSegmentedRadixSort::SortPairsDescending(
  //     workspace5, workspace_bytes5, _unique_frequency, _unique_frequency,
  //     _unique_dst, _unique_dst, _num_unique, num_input_node, num_edge_prefix,
  //     num_edge_prefix + 1, 0, sizeof(IdType) * 8, cu_stream));
  // device->StreamSync(_ctx, stream);

  size_t workspace_bytes5;
  Id64Type *key_out = static_cast<Id64Type *>(
      device->AllocDataSpace(_ctx, _unique_list_size * sizeof(Id64Type)));
  IdType *val_out = static_cast<IdType *>(
      device->AllocDataSpace(_ctx, _unique_list_size * sizeof(IdType)));
  CUDA_CALL(cub::DeviceRadixSort::SortPairsDescending(
      nullptr, workspace_bytes5, static_cast<Id64Type *>(nullptr),
      static_cast<Id64Type *>(nullptr), static_cast<IdType *>(nullptr),
      static_cast<IdType *>(nullptr), _num_unique, 0, sizeof(Id64Type) * 8,
      cu_stream));
  device->StreamSync(_ctx, stream);

  // validate_combine_w_freq<Constant::kCudaBlockSize, Constant::kCudaTileSize>
  //     <<<grid3, block3, 0, cu_stream>>>(_unique_src, _unique_frequency,
  //                                       _unique_combination_key, _num_unique);
  // device->StreamSync(_ctx, stream);
  void *workspace5 = device->AllocWorkspace(_ctx, workspace_bytes5);
  CUDA_CALL(cub::DeviceRadixSort::SortPairsDescending(
      workspace5, workspace_bytes5, _unique_combination_key,
      key_out, _unique_dst, val_out, _num_unique, 0,
      sizeof(Id64Type) * 8, cu_stream));
  device->StreamSync(_ctx, stream);
  // validate_combine<Constant::kCudaBlockSize, Constant::kCudaTileSize>
  //     <<<grid3, block3, 0, cu_stream>>>(_unique_src, key_out, _num_unique);
  // device->StreamSync(_ctx, stream);
  device->FreeDataSpace(_ctx, _unique_combination_key);
  device->FreeDataSpace(_ctx, _unique_dst);
  _unique_combination_key = key_out;
  _unique_dst = val_out;
  double step8_time = t8.Passed();
  LOG(DEBUG) << "FrequencyHashmap::GetTopK step 8 finish";

  // 9. prefix the number of output edges for each nodes that we get in step 7
  Timer t9;
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(workspace2, workspace_bytes2,
                                          num_output_prefix, num_output_prefix,
                                          num_input_node + 1, cu_stream));
  device->StreamSync(_ctx, stream);
  double step9_time = t9.Passed();
  LOG(DEBUG) << "FrequencyHashmap::GetTopK step 9 finish";

  // 10. copy the result to the output array and set the value of num_output
  /** SXN: bug: unique_frequency is not sorted according to unique_dst*/
  Timer t10;
  compact_output<<<grid2, block2, 0, cu_stream>>>(
      _unique_src, _unique_dst, _unique_frequency, num_input_node, K,
      num_edge_prefix, num_output_prefix, output_src, output_dst, output_data,
      num_output);
  device->StreamSync(_ctx, stream);

  double step10_time = t10.Passed();
  LOG(DEBUG) << "FrequencyHashmap::GetTopK step 10 finish";

  // 11. reset data
  Timer t11;
  reset_node_table<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid0, block0, 0, cu_stream>>>(device_table, _node_list, _num_node);
  Device::Get(_ctx)->StreamSync(_ctx, stream);

  reset_edge_table<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid3, block3, 0, cu_stream>>>(device_table, _unique_node_idx,
                                        _unique_dst, _num_unique);
  Device::Get(_ctx)->StreamSync(_ctx, stream);
  double step11_time = t11.Passed();

  LOG(DEBUG) << "FrequencyHashmap::GetTopK step 11 finish";

  _num_node = 0;
  _num_unique = 0;

  device->FreeWorkspace(_ctx, workspace5);
  device->FreeWorkspace(_ctx, num_output_prefix);
  device->FreeWorkspace(_ctx, num_edge_prefix);
  device->FreeWorkspace(_ctx, workspace4);
  device->FreeWorkspace(_ctx, workspace3);
  device->FreeWorkspace(_ctx, unique_idx);
  device->FreeWorkspace(_ctx, tmp_unique_frequency);
  device->FreeWorkspace(_ctx, tmp_unique_dst);
  device->FreeWorkspace(_ctx, tmp_unique_node_idx);
  device->FreeWorkspace(_ctx, device_num_unique);
  device->FreeWorkspace(_ctx, num_unique_prefix);
  device->FreeWorkspace(_ctx, workspace2);
  device->FreeWorkspace(_ctx, workspace1);
  device->FreeWorkspace(_ctx, workspace0);

  Profiler::Get().LogStepAdd(task_key, kLogL3RandomWalkTopKStep1Time,
                             step1_time);
  Profiler::Get().LogStepAdd(task_key, kLogL3RandomWalkTopKStep2Time,
                             step2_time);
  Profiler::Get().LogStepAdd(task_key, kLogL3RandomWalkTopKStep3Time,
                             step3_time);
  Profiler::Get().LogStepAdd(task_key, kLogL3RandomWalkTopKStep4Time,
                             step4_time);
  Profiler::Get().LogStepAdd(task_key, kLogL3RandomWalkTopKStep5Time,
                             step5_time);
  Profiler::Get().LogStepAdd(task_key, kLogL3RandomWalkTopKStep6Time,
                             step6_time);
  Profiler::Get().LogStepAdd(task_key, kLogL3RandomWalkTopKStep7Time,
                             step7_time);
  Profiler::Get().LogStepAdd(task_key, kLogL3RandomWalkTopKStep8Time,
                             step8_time);
  Profiler::Get().LogStepAdd(task_key, kLogL3RandomWalkTopKStep9Time,
                             step9_time);
  Profiler::Get().LogStepAdd(task_key, kLogL3RandomWalkTopKStep10Time,
                             step10_time);
  Profiler::Get().LogStepAdd(task_key, kLogL3RandomWalkTopKStep11Time,
                             step11_time);
}

#else
void FrequencyHashmap::GetTopK(
    IdType *input_src, IdType *input_dst,
    const size_t num_input_edge, const IdType *input_nodes,
    const size_t num_input_node, const size_t K, IdType *output_src,
    IdType *output_dst, IdType *output_data, size_t *num_output,
    StreamHandle stream, uint64_t task_key) {
  const size_t num_tiles0 = RoundUpDiv(num_input_node, Constant::kCudaTileSize);
  const size_t num_tiles1 = RoundUpDiv(num_input_edge, Constant::kCudaTileSize);
  const dim3 grid_input_node(num_tiles0);
  const dim3 grid_input_edge(num_tiles1);

  const dim3 block_input_node(Constant::kCudaBlockSize);
  const dim3 block_input_edge(Constant::kCudaBlockSize);
  dim3 block2(Constant::kCudaBlockSize, 1);
  while (static_cast<size_t>(block2.x) >= 2 * K) {
    block2.x /= 2;
    block2.y *= 2;
  }
  dim3 grid2(RoundUpDiv(num_input_node, static_cast<size_t>(block2.y)));

  auto device_table = MutableDeviceFrequencyHashmap(this);
  auto device = Device::Get(_ctx);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  size_t workspace_bytes1;
  size_t workspace_bytes2;
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      nullptr, workspace_bytes1, static_cast<IdType *>(nullptr),
      static_cast<IdType *>(nullptr), grid_input_edge.x + 1, cu_stream));
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      nullptr, workspace_bytes2, static_cast<IdType *>(nullptr),
      static_cast<IdType *>(nullptr), num_input_node + 1, cu_stream));
  device->StreamSync(_ctx, stream);

  void *workspace1 = device->AllocWorkspace(_ctx, workspace_bytes1);
  void *workspace2 = device->AllocWorkspace(_ctx, workspace_bytes2);

  // 1. populate the node table
  Timer t1;
  populate_node_table<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid_input_node, block_input_node, 0, cu_stream>>>(input_nodes, num_input_node,
                                        device_table);
  device->StreamSync(_ctx, stream);
  double step1_time = t1.Passed();

  LOG(DEBUG) << "FrequencyHashmap::GetTopK step 1 finish with "
             << num_input_node << " input nodes with grid " << grid_input_node.x
             << " block " << block_input_node.x;

  // 2. count frequency of every unique edge and
  //    count unique edges for every node
  Timer t2;
  IdType *num_unique_prefix = static_cast<IdType *>(
      device->AllocWorkspace(_ctx, sizeof(IdType) * (grid_input_edge.x + 1)));
  count_frequency_revised<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid_input_edge, block_input_edge, 0, cu_stream>>>(input_src, input_dst, num_input_edge,
                                        _edges_per_node, num_unique_prefix,
                                        device_table);
  device->StreamSync(_ctx, stream);
  /** pos in device_table is now stored in input_src */
  double step2_time = t2.Passed();
  LOG(DEBUG) << "FrequencyHashmap::GetTopK step 2 finish";

  // 3. count the number of unique edges.
  //    prefix sum the the array
  Timer t3;

  CUDA_CALL(cub::DeviceScan::ExclusiveSum(workspace1, workspace_bytes1,
                                          num_unique_prefix, num_unique_prefix,
                                          grid_input_edge.x + 1, cu_stream));
  device->StreamSync(_ctx, stream);
  double step3_time = t3.Passed();
  LOG(DEBUG) << "FrequencyHashmap::GetTopK step 3 finish";

  // 4. get the array of all unique edges' pos in table
  Timer t4;
  device->CopyDataFromTo(&num_unique_prefix[grid_input_edge.x], 0, &_num_unique, 0,
                         sizeof(IdType), _ctx, CPU(), stream);
  device->StreamSync(_ctx, stream);
  LOG(DEBUG) << "FrequencyHashmap::Before gettopk step 4,  number of unique is " << _num_unique;
  /** location in edge table */
  /** now we reuse input_dst as pos */
  IdType *tmp_unique_pos = input_dst;
  input_dst = nullptr;
  generate_unique_edges_pos<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid_input_edge, block_input_edge, 0, cu_stream>>>(input_src, input_nodes, num_input_node, num_input_edge,
                                        num_unique_prefix, _unique_combination_key, tmp_unique_pos,
                                        _edges_per_node, device_table);
  device->StreamSync(_ctx, stream);
  LOG(DEBUG) << "FrequencyHashmap::GetTopK step 4 finish with number of unique "
             << _num_unique;
  double step4_time = t4.Passed();

  // 5. pair-sort unique array using src as key
  //    construct the array of unique dst, unique node idx
  Timer t5;
  /** now we reuse input_src as sort tmp space */
  IdType *alt_val = input_src;
  input_src = nullptr;
  cub::DoubleBuffer<Id64Type> keys(_unique_combination_key, reinterpret_cast<Id64Type*>(_unique_node_idx));
  cub::DoubleBuffer<IdType>   vals(tmp_unique_pos, alt_val);

  size_t workspace_bytes4;
  CUDA_CALL(cub::DeviceRadixSort::SortPairsDescending(
      nullptr, workspace_bytes4, keys, vals, _num_unique, 0, sizeof(Id64Type) * 8,
      cu_stream));
  device->StreamSync(_ctx, stream);

  void *workspace4 = device->AllocWorkspace(_ctx, workspace_bytes4);
  CUDA_CALL(cub::DeviceRadixSort::SortPairsDescending(
      workspace4, workspace_bytes4, keys, vals, _num_unique, 0, sizeof(IdType) * 8,
      cu_stream));
  device->StreamSync(_ctx, stream);

  _unique_combination_key = keys.Current();
  tmp_unique_pos = vals.Current();
  _unique_node_idx = reinterpret_cast<IdType*>(keys.Alternate());
  _unique_dst = _unique_node_idx + _unique_list_size;
  alt_val = vals.Alternate();

  const size_t num_tiles3 = RoundUpDiv(_num_unique, Constant::kCudaTileSize);
  const dim3 grid_uniq_e(num_tiles3);
  const dim3 block_uniq_e(Constant::kCudaBlockSize);
  construct_unique_edge_list<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid_uniq_e, block_uniq_e, 0, cu_stream>>>(
          _unique_combination_key, tmp_unique_pos,
          _unique_node_idx, _unique_dst, _num_unique, device_table);
  device->StreamSync(_ctx, stream);
  double step5_time = t5.Passed();
  LOG(DEBUG) << "FrequencyHashmap::GetTopK step 5 finish";


  // 6. sort the unique src node array.
  Timer t6;
  size_t workspace_bytes6;
  CUDA_CALL(cub::DeviceRadixSort::SortKeysDescending(
      nullptr, workspace_bytes6, static_cast<IdType *>(nullptr),
      static_cast<IdType *>(nullptr), num_input_node, 0, sizeof(IdType) * 8,
      cu_stream));
  device->StreamSync(_ctx, stream);

  void *workspace6 = device->AllocWorkspace(_ctx, workspace_bytes6);
  // CUDA_CALL(cub::DeviceRadixSort::SortKeysDescending(
  //     workspace6, workspace_bytes6, input_nodes, _node_list, num_input_node, 0,
  //     sizeof(IdType) * 8, cu_stream));
  // device->StreamSync(_ctx, stream);

  double step6_time = t6.Passed();
  LOG(DEBUG) << "FrequencyHashmap::GetTopK step 6 finish";

  // 7. get array unique edge number in the order of src nodes.
  //    also count the number of output edges for each nodes.
  //    prefix sum for array of unique edge number.
  Timer t7;
  IdType *num_edge_prefix = static_cast<IdType *>(
      device->AllocWorkspace(_ctx, (num_input_node + 1) * sizeof(IdType)));
  IdType *num_output_prefix = static_cast<IdType *>(
      device->AllocWorkspace(_ctx, (num_input_node + 1) * sizeof(IdType)));
  generate_num_edge<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid_input_node, block_input_node, 0, cu_stream>>>(input_nodes, num_input_node, K,
                                        num_edge_prefix, num_output_prefix,
                                        device_table);
  device->StreamSync(_ctx, stream);

  CUDA_CALL(cub::DeviceScan::ExclusiveSum(workspace2, workspace_bytes2,
                                          num_edge_prefix, num_edge_prefix,
                                          num_input_node + 1, cu_stream));
  device->StreamSync(_ctx, stream);

  CUDA_CALL(cub::DeviceScan::ExclusiveSum(workspace2, workspace_bytes2,
                                          num_output_prefix, num_output_prefix,
                                          num_input_node + 1, cu_stream));
  device->StreamSync(_ctx, stream);
  double step7_time = t7.Passed();
  LOG(DEBUG) << "FrequencyHashmap::GetTopK step 7 finish";

  // 8. nothing to be done
  Timer t8;
  double step8_time = t8.Passed();
  LOG(DEBUG) << "FrequencyHashmap::GetTopK step 8 finish";


  // 9. nothing here
  Timer t9;
  double step9_time = t9.Passed();
  LOG(DEBUG) << "FrequencyHashmap::GetTopK step 9 finish";

  // 10. copy the result to the output array and set the value of num_output
  /** SXN: bug: unique_frequency is not sorted according to unique_dst*/
  Timer t10;
  compact_output_revised<<<grid2, block2, 0, cu_stream>>>(
      input_nodes,
      _unique_combination_key, _unique_dst, num_input_node, K,
      num_edge_prefix, num_output_prefix, output_src, output_dst, output_data,
      num_output);
  device->StreamSync(_ctx, stream);

  double step10_time = t10.Passed();
  LOG(DEBUG) << "FrequencyHashmap::GetTopK step 10 finish";

  // 11. reset data
  Timer t11;
  reset_node_table<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid_input_node, block_input_node, 0, cu_stream>>>(device_table, input_nodes, num_input_node);
  Device::Get(_ctx)->StreamSync(_ctx, stream);

  reset_edge_table_revised<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid_uniq_e, block_uniq_e, 0, cu_stream>>>(device_table, tmp_unique_pos,
                                        _unique_dst, _num_unique);
  Device::Get(_ctx)->StreamSync(_ctx, stream);
  double step11_time = t11.Passed();

  LOG(DEBUG) << "FrequencyHashmap::GetTopK step 11 finish";

  _num_unique = 0;

  device->FreeWorkspace(_ctx, num_output_prefix);
  device->FreeWorkspace(_ctx, num_edge_prefix);
  device->FreeWorkspace(_ctx, workspace6);
  device->FreeWorkspace(_ctx, workspace4);
  device->FreeWorkspace(_ctx, num_unique_prefix);
  device->FreeWorkspace(_ctx, workspace2);
  device->FreeWorkspace(_ctx, workspace1);

  Profiler::Get().LogStepAdd(task_key, kLogL3RandomWalkTopKStep1Time,
                             step1_time);
  Profiler::Get().LogStepAdd(task_key, kLogL3RandomWalkTopKStep2Time,
                             step2_time);
  Profiler::Get().LogStepAdd(task_key, kLogL3RandomWalkTopKStep3Time,
                             step3_time);
  Profiler::Get().LogStepAdd(task_key, kLogL3RandomWalkTopKStep4Time,
                             step4_time);
  Profiler::Get().LogStepAdd(task_key, kLogL3RandomWalkTopKStep5Time,
                             step5_time);
  Profiler::Get().LogStepAdd(task_key, kLogL3RandomWalkTopKStep6Time,
                             step6_time);
  Profiler::Get().LogStepAdd(task_key, kLogL3RandomWalkTopKStep7Time,
                             step7_time);
  Profiler::Get().LogStepAdd(task_key, kLogL3RandomWalkTopKStep8Time,
                             step8_time);
  Profiler::Get().LogStepAdd(task_key, kLogL3RandomWalkTopKStep9Time,
                             step9_time);
  Profiler::Get().LogStepAdd(task_key, kLogL3RandomWalkTopKStep10Time,
                             step10_time);
  Profiler::Get().LogStepAdd(task_key, kLogL3RandomWalkTopKStep11Time,
                             step11_time);
}
#endif
}  // namespace cuda
}  // namespace common
}  // namespace samgraph