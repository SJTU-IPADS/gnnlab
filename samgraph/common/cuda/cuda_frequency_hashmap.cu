#include <cuda_runtime.h>

#include <cub/cub.cuh>

#include "../constant.h"
#include "../device.h"
#include "../logging.h"
#include "cuda_frequency_hashmap.h"
#include "cuda_utils.h"

namespace samgraph {
namespace common {
namespace cuda {

namespace {

size_t TableSize(const size_t num, const size_t scale) {
  const size_t next_pow2 = 1 << static_cast<size_t>(1 + std::log2(num >> 1));
  return next_pow2 << scale;
}

class MutableDeviceFrequencyHashmap : public DeviceFrequencyHashmap {
 public:
  typedef typename DeviceFrequencyHashmap::NodeBucket *NodeIterator;
  typedef typename DeviceFrequencyHashmap::EdgeBucket *EdgeIterator;

  explicit MutableDeviceFrequencyHashmap(FrequencyHashmap *const host_map)
      : DeviceFrequencyHashmap(host_map->DeviceHandle()) {}

  inline __device__ bool AttemptInsertNodeAt(const IdType pos,
                                             const IdType id) {
    NodeIterator bucket = GetMutableNode(pos);
    const IdType key = atomicCAS(&bucket->key, kEmptyKey, id);
    if (key == Constant::kEmptyKey || key == id) {
      return true;
    } else {
      return false;
    }
  }

  inline __device__ NodeIterator InsertOrGetNode(const IdType id) {
    IdType pos = HashNode(id);

    IdType delta = 1;
    while (!AttemptInsertNodeAt(pos, id)) {
      pos = HashNode(pos + delta);
      delta += 1;
    }
    return GetMutableNode(pos);
  }

  inline __device__ bool AttemptInsertEdgeAt(const LongIdType pos,
                                             const LongIdType edge_id,
                                             const IdType node_id,
                                             const IdType index) {
    EdgeIterator bucket = GetMutableEdge(pos);
    const LongIdType key = atomicCAS(&bucket->key, kEmptyLongKey, edge_id);
    if (key == Constant::kEmptyLongKey || key == edge_id) {
      atomicAdd(&bucket->count, 1LLU);
      atomicMin(&bucket->index, index);
      if (key == Constant::kEmptyKey) {
        NodeIterator node = InsertOrGetNode(node_id);
        atomicAdd(&bucket->count, 1U);
      }
      return true;
    } else {
      return false;
    }
  }

  inline __device__ EdgeIterator InsertEdge(const IdType src, const IdType dst,
                                            const IdType index) {
    LongIdType id = EncodeEdge(src, dst);
    LongIdType pos = HashEdge(id);

    LongIdType delta = 1;
    while (!AttemptInsertEdgeAt(pos, id, src, index)) {
      pos = HashEdge(pos + delta);
      delta += 1;
    }

    return GetMutableEdge(pos);
  }

  inline __device__ NodeIterator GetMutableNode(const IdType pos) {
    assert(pos < _ntable_size);
    return const_cast<NodeIterator>(_node_table + pos);
  }

  inline __device__ EdgeIterator GetMutableEdge(const LongIdType pos) {
    assert(pos < _etable_size);
    return const_cast<EdgeIterator>(_edge_table + pos);
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
  for (LongIdType index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_bucket) {
      EdgeIterator edge_iter = table.GetMutableEdge(index);
      edge_iter->key = Constant::kEmptyLongKey;
      edge_iter->count = 0;
    }
  }
}

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void reset_node_table(MutableDeviceFrequencyHashmap table,
                                 IdType *nodes, const size_t num_nodes) {
  assert(BLOCK_SIZE == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  using NodeIterator = typename MutableDeviceFrequencyHashmap::NodeIterator;

#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_nodes) {
      IdType id = nodes[index];
      IdType pos = table.SearchNodeForPosition(id);
      NodeIterator node_iter = table.GetMutableNode(pos);
      node_iter->key = Constant::kEmptyKey;
      node_iter->count = 0;
    }
  }
}
}

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void reset_edge_table(MutableDeviceFrequencyHashmap table,
                                 IdType *unique_src, IdType *unique_dst,
                                 const size_t num_unique) {
  assert(BLOCK_SIZE == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  using EdgeIterator = typename MutableDeviceFrequencyHashmap::EdgeIterator;

#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_unique) {
      IdType src = unique_src[index];
      IdType dst = unique_dst[index];
      LongIdType pos = table.SearchEdgeForPosition(src, dst);
      EdgeIterator edge_iter = table.GetMutableEdge(pos);
      edge_iter->key = Constant::kEmptyLongKey;
      edge_iter->count = 0;
    }
  }
}

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void count_frequency(const IdType *input_src,
                                const IdType *input_dst,
                                const size_t num_input_edge,
                                MutableDeviceFrequencyHashmap table) {
  assert(BLOCK_SIZE == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_input_edge && input_src != Constant::kEmptyKey) {
      table.InsertEdge(input_src[index], input_dst[index], index);
    }
  }
}

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void count_unique_edges(const IdType *input_src,
                                   const IdType *input_dst,
                                   const size_t num_input_edge,
                                   IdType *item_prefix,
                                   DeviceFrequencyHashmap table) {
  assert(BLOCK_SIZE == blockDim.x);

  using BlockReduce = typename cub::BlockReduce<IdType, BLOCK_SIZE>;
  using EdgeBucket = typename DeviceFrequencyHashmap::EdgeBucket;

  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  IdType count = 0;
#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_input_edge) {
      const EdgeBucket &bucket =
          *table.SearchEdge(input_src[index], input_dst[index]);
      if (bucket.index == index) {
        ++count;
      }
    }
  }

  __shared__ typename BlockReduce::TempStorage temp_storage;

  count = BlockReduce(temp_space).Sum(count);

  if (threadIdx.x == 0) {
    item_prefix[blockIdx.x] = count;
    if (blockIdx.x == 0) {
      item_prefix[gridDim.x] = 0;
    }
  }
}

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void generate_unique_edges(const IdType *input_src,
                                      const IdType *input_dst,
                                      const size_t num_input_edge,
                                      IdType *item_prefix, IdType *unique_src,
                                      IdType *unique_dst, size_t *num_unique,
                                      DeviceFrequencyHashmap table) {
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

    FlagType flag;
    EdgeBucket *bucket;
    if (index < num_input_edge) {
      bucket = table.SearchEdge(input_src[index], input_dst[index]);
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
      unique_src[pos] = input_src[index];
      unique_dst[pos] = input_dst[index];
    }
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *num_unique = item_prefix[gridDim.x];
  }
}

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void generate_unique_edge_frequency(const IdType *unique_src,
                                               const IdType *unique_dst,
                                               IdType *unique_count,
                                               const size_t num_unique,
                                               DeviceFrequencyHashmap table) {
  assert(BLOCK_SIZE == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  using EdgeBucket = typename DeviceFrequencyHashmap::EdgeBucket;

#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_unique) {
      const EdgeBucket &bucket =
          *table.SearchEdge(unique_src[index], unique_dst[index]);
      unique_count[index] = bucket.count;
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
                               const IdType *unique_dst, const size_t num_nodes,
                               const size_t K, const IdType *num_unique_prefix,
                               const IdType *num_output_prefix,
                               IdType *output_src, IdType *output_dst,
                               size_t *num_output) {
  size_t i = blockIdx.x * blockDim.y + threadIdx.y;
  const size_t stride = blockDim.y * gridDim.x;

  while (i < num_nodes) {
    IdType k = threadIdx.y;
    while (k < K) {
      IdType from_off = num_unique_prefix[index] + k;
      IdType to_off = num_unique_prefix[index] + k;

      output_src[to_off] = unique_src[from_off];
      output_dst[to_off] = unique_dst[from_off];

      k += blockDim.x;
    }

    i += stride;
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *num_output = num_output_prefix[num_nodes];
  }
}

}  // namespace

FrequencyHashmap(const size_t max_nodes, const size_t max_edges, Context ctx,
                 StreamHandle stream, const size_t scale)
    : _ctx(ctx),
      _ntable_size(TableSize(max_nodes, scale)),
      _etable_size(TableSize(max_edges, scale)),
      _num_nodes(0),
      _node_list_size(max_nodes),
      _num_unique(0),
      _unique_list_size(max_edges) {
  auto device = Device::Get(_ctx);
  CHECK_EQ(device.device_type, kGPU);

  _node_table = static_cast<NodeBucket *>(
      device->AllocDataSpace(_ctx, sizeof(NodeBucket) * _ntable_size));
  _edge_table = static_cast<EdgeBucket *>(
      device->AllocDataSpace(_ctx, sizeof(EdgeBucket) * _etable_size));

  _node_list = static_cast<IdType *>(
      device->AllocDataSpace(_ctx, sizeof(IdType) * _node_list_size));

  _unique_src = static_cast<IdType *>(
      device->AllocDataSpace(_ctx, sizeof(IdType) * _unique_list_size));
  _unique_dst = static_cast<IdType *>(
      device->AllocDataSpace(_ctx, sizeof(IdType) * _unique_list_size));
  _unique_frequency = static_cast<IdType *>(
      device->AllocDataSpace(_ctx, sizeof(IdType) * _unique_list_size));

  auto device_table = MutableDeviceFrequencyHashmap(this);
  dim3 grid0(RoundUpDiv(_node_list_size, kCudaTileSize));
  dim3 grid1(RoundUpDiv(_unique_list_size, kCudaTileSize));
  dim3 block0(Constant::kCudaBlockSize);
  dim3 block1(Constant::kCudaBlockSize);

  init_node_table<<<grid0, block0>>>(device_table, _node_list_size);
  init_edge_table<<<grid1, block1>>>(device_tablem, _unique_list_size);
}

FrequencyHashmap::~FrequencyHashmap() {
  auto device = Device::Get(_ctx);

  device->FreeDataSpace(_node_table);
  device->FreeDataSpace(_edge_table);
  device->FreeDataSpace(_node_list);
  device->FreeDataSpace(_unique_src);
  device->FreeDataSpace(_unique_dst);
  device->FreeDataSpace(_unique_frequency);
}

void FrequencyHashmap::GetTopK(const IdType *input_src, const IdType *input_dst,
                               const size_t num_input_edge,
                               const IdType *input_nodes,
                               const size_t num_input_node, const size_t K,
                               IdType *output_src, IdType *output_dst,
                               size_t *num_output, StreamHandle stream) {
  const size_t num_tiles0 = RoundUpDiv(num_input_edge, Constant::kCudaTileSize);
  const size_t num_tiles1 = RoundUpDiv(num_input_node, Constant::kCudaTileSize);
  const dim3 grid0(num_tiles0);
  const dim3 grid1(num_tiles1);

  const dim3 block0(Constant::kCudaBlockSize);
  const dim3 block1(Constant::kCudaBlockSize);

  const dim3 block2(Constant::kCudaBlockSize, 1);
  while (static_cast<size_t>(block2.x) >= 2 * K) {
    block2.x /= 2;
    block2.y *= 2;
  }
  const dim3 grid2(RoundUpDiv(num_input_node, static_cast<size_t>(block2.y));

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

  // 1. count frequency of every unique edge and
  //    count unique edges for every node
  count_frequency<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid0, block0, 0, cu_stream>>>(input_src, input_dst, num_input_edge,
                                        device_table);
  device->StreamSync(_ctx, stream);

  // 2. count the number of unique edges.
  //    prefix sum the the array
  unique_edge_number_prefix = static_cast<IdType *>(
      device->AllocWorkspace(_ctx, sizeof(IdType) * (grid0.x + 1)));
  count_unique_edges<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid0, block0, 0, cu_stream>>>(input_src, input_dst, num_input,
                                        unique_edge_number_prefix,
                                        device_table);
  device->StreamSync(_ctx, stream);

  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      workspace0, workspace_bytes0, unique_edge_number_prefix,
      unique_edge_number_prefix, grid0.x + 1, cu_stream));
  device->StreamSync(_ctx, stream);

  // 3. get the array of all unique edges.
  size_t *device_num_unique =
      static_cast<size_t *>(device->AllocWorkspace(_ctx, sizeof(size_t)));
  generate_unique_edges<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid0, block0, 0, cu_stream>>>(
          input_src, input_dst, num_input, unique_edge_number_prefix,
          _unique_src, _unique_dst, device_num_unique, device_table);
  device->StreamSync(_ctx, stream);

  device->CopyDataFromTo(device_num_unique, 0, _num_unique, 0, sizeof(size_t),
                         _ctx, CPU(), stream);
  device->StreamSync(_ctx, stream);

  // 4. pair-sort unique edges(src, dst) using src as key.
  size_t workspace_bytes3;
  CUDA_CALL(cub::DeviceRadixSort::SortPairs(
      nullptr, workspace_bytes3, static_cast<IdType *>(nullptr),
      static_cast<IdType *>(nullptr), static_cast<IdType *>(nullptr),
      static_cast<IdType *>(nullptr), _num_unique, 0, sizeof(IdType) * 8,
      cu_stream));
  device->StreamSync(_ctx, stream);

  void *workspace3 = device->AllocWorkspace(_ctx, workspace_bytes3);
  CUDA_CALL(cub::DeviceRadixSort::SortPairs(
      workspace3, workspace_bytes3, _unique_src, _unique_src, _unique_dst,
      _unique_dst, _num_unique, 0, sizeof(IdType) * 8, cu_stream));
  device->StreamSync(_ctx, stream);

  // 5. get the frequency array in the order of sorted edge.
  const size_t num_tiles2 = RoundUpDiv(_num_unique, Constant::kCudaTileSize);
  const dim3 grid3(num_tiles2);
  const dim3 block3(Constant::kCudaBlockSize);
  generate_unique_edge_frequency<Constant::kCudaBlockSize,
                                 Constant::kCudaTileSize>
      <<<grid3, block3, 0, cu_stream>>>(_unique_src, _unique_dst, _unique_count,
                                        _num_unique, device_table);
  device->StreamSync(_ctx, stream);

  // 6. sort the unique src node array.
  IdType *sorted_nodes =
      static_cast<IdType *>(device->AllocWorkspace(_ctx, num_input_node));
  device->CopyDataFromTo(input_nodes, 0, sorted_nodes, 0,
                         num_input_node * sizeof(IdType), _ctx, _ctx, stream);
  device->StreamSync(_ctx, stream);

  size_t workspace_bytes4;
  CUDA_CALL(cub::DeviceRadixSort::SortKeys(
      nullptr, workspace_bytes4, static_cast<IdType *>(nullptr),
      static_cast<IdType *>(nullptr), num_input_node, 0, sizeof(IdType) * 8,
      cu_stream));
  device->StreamSync(_ctx, stream);

  void *workspace4 = device->AllocWorkspace(_ctx, workspace_bytes4);
  CUDA_CALL(cub::DeviceRadixSort::SortKeys(
      workspace4, workspace_bytes4, sorted_nodes, sorted_nodes, num_input_node,
      0, sizeof(IdType) * 8, cu_stream));
  device->StreamSync(_ctx, stream);

  // 7. get array unique edge number in the order of src nodes.
  //    also count the number of output edges for each nodes.
  //    prefix sum for array of unique edge number.
  IdType *num_edge_prefix = static_cast<IdType *>(
      device->AllocWorkspace(_ctx, (num_input_node + 1) * sizeof(IdType)));
  IdType *num_output_prefix = static_cast<IdType *>(
      device->AllocWorkspace(_ctx, (num_input_node + 1) * sizeof(IdType)));
  generate_num_edge<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid1, block1, 0, cu_stream>>>(sorted_nodes, num_input_node, K,
                                        num_edge_prefix, num_output_prefix,
                                        device_table);
  device->StreamSync(_ctx, stream);

  CUDA_CALL(cub::DeviceScan::ExclusiveSum(workspace2, workspace_bytes2,
                                          num_edge_prefix, num_edge_prefix,
                                          num_input_node + 1, cu_stream));
  device->StreamSync(_ctx, stream);

  // 8. segment-sort the edge for every node using the frequency as key
  //    and the dst as value.
  size_t workspace_bytes5;
  CUDA_CALL(cub::DeviceSegmentedRadixSort::SortPairsDescending(
      nullptr, workspace_bytes5, static_cast<IdType *>(nullptr),
      static_cast<IdType *>(nullptr), static_cast<IdType *>(nullptr),
      static_cast<IdType *>(nullptr), _num_unique, num_input_node,
      static_cast<IdType *>(nullptr), static_cast<IdType *>(nullptr), 0,
      sizeof(IdType) * 8, cu_stream));
  device->StreamSync(_ctx, stream);

  void workspace5 = device->AllocWorkspace(_ctx, workspace_bytes5);
  CUDA_CALL(cub::DeviceSegmentedRadixSort::SortPairsDescending(
      workspace5, workspace_bytes5, _unique_count, _unique_count, _unique_dst,
      _unique_dst, _num_unique, num_input_node, num_edge_prefix,
      num_edge_prefix + 1, sizeof(IdType) * 8, cu_stream));
  device->StreamSync(_ctx, stream);

  // 9. prefix the number of output edges for each nodes that we get in step 7
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(workspace2, workspace_bytes2,
                                          num_output_prefix, num_output_prefix,
                                          num_input_node + 1, cu_stream));
  device->StreamSync(_ctx, stream);

  // 10. copy the result to the output array and set the value of num_output
  device_num_output =
      static_cast<IdType *>(device->AllocWorkspace(_ctx, sizeof(IdType)));
  compact_output<<<grid2, block2, 0, cu_stream>>>(_unique_src,
_unique_dst, num_input_node,
    K, num_unique_prefix,
    num_output_prefix,
    output_src, output_dst,
    device_num_output);
  device->StreamSync(_ctx, stream);

  device->CopyDataFromTo(device_num_output, 0, num_output, 0, sizeof(size_t),
                         _ctx, CPU(), stream);
  device->StreamSync(_ctx, stream);

  device->FreeWorkspace(_ctx, device_num_output);
  device->FreeWorkspace(_ctx, workspace5);
  device->FreeWorkspace(_ctx, num_output_prefix);
  device->FreeWorkspace(_ctx, num_edge_prefix);
  device->FreeWorkspace(_ctx, workspace4);
  device->FreeWorkspace(_ctx, sorted_nodes);
  device->FreeWorkspace(_ctx, workspace3);
  device->FreeWorkspace(_ctx, device_num_unique);
  device->FreeWorkspace(_ctx, unique_edge_number_prefix);
  device->FreeWorkspace(_ctx, workspace2);
  device->FreeWorkspace(_ctx, workspace1);
  device->FreeWorkspace(_ctx, workspace0);
}

void FrequencyHashmap::Reset(StreamHandle stream) {
  auto device_table = MutableDeviceFrequencyHashmap(this);
  auto device = Device::Get(_ctx);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  dim3 grid0(RoundUpDiv(_num_nodes, kCudaTileSize));
  dim3 grid1(RoundUpDiv(_num_unique, kCudaTileSize));

  dim3 block0(Constant::kCudaBlockSize);
  dim3 block1(Constant::kCudaBlockSize);

  reset_node_table<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid0, block0>>>(device_table, _sorted_nodes, _num_nodes);
  Device::Get(_ctx)->StreamSync(_ctx, stream);

  reset_edge_table<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid1, block1>>>(device_table, _unique_src, _unique_dst, num_unique);
  Device::Get(_ctx)->StreamSync(_ctx, stream);

  _num_unique = 0;
}

DeviceFrequencyHashmap FrequencyHashmap::DeviceHandle() {
  return DeviceFrequencyHashmap(_node_table, _edge_table, _ntable_size,
                                _etable_size, _unique_src, _unique_dst,
                                _unique_count, _num_unique_edge);
}

}  // namespace cuda
}  // namespace common
}  // namespace samgraph