#include <curand.h>
#include <curand_kernel.h>

#include <cassert>
#include <chrono>
#include <cstdio>
#include <cub/cub.cuh>

#include "../common.h"
#include "../constant.h"
#include "../device.h"
#include "../logging.h"
#include "../profiler.h"
#include "../timer.h"
#include "../partition.h"
#include "../run_config.h"
#include "cuda_function.h"
#include "cuda_utils.h"

#define NEW_ALGO

namespace samgraph {
namespace common {
namespace cuda {

namespace {

#ifndef NEW_ALGO
template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void sample_khop0(const IdType *indptr, const IdType *indices,
                             const IdType *input, const size_t num_input,
                             const size_t fanout, IdType *tmp_src,
                             IdType *tmp_dst, curandState *random_states,
                             size_t num_random_states) {
  assert(BLOCK_SIZE == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  assert(i < num_random_states);
  curandState local_state = random_states[i];

  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_input) {
      const IdType rid = input[index];
      const IdType off = indptr[rid];
      const IdType len = indptr[rid + 1] - indptr[rid];

      if (len <= fanout) {
        size_t j = 0;
        for (; j < len; ++j) {
          tmp_src[index * fanout + j] = rid;
          tmp_dst[index * fanout + j] = indices[off + j];
        }

        for (; j < fanout; ++j) {
          tmp_src[index * fanout + j] = Constant::kEmptyKey;
          tmp_dst[index * fanout + j] = Constant::kEmptyKey;
        }
      } else {
        for (size_t j = 0; j < fanout; ++j) {
          tmp_src[index * fanout + j] = rid;
          tmp_dst[index * fanout + j] = indices[off + j];
        }

        for (size_t j = fanout; j < len; ++j) {
          size_t k = curand(&local_state) % (j + 1);
          if (k < fanout) {
            tmp_dst[index * fanout + k] = indices[off + j];
          }
        }
      }
    }
  }

  random_states[i] = local_state;
}

#else

template <size_t WARP_SIZE, size_t BLOCK_WARP, size_t TILE_SIZE>
__global__ void sample_khop0(const IdType *indptr, const IdType *indices,
                             const IdType *input, const size_t num_input,
                             const size_t fanout, IdType *tmp_src,
                             IdType *tmp_dst, curandState *random_states,
                             size_t num_random_states) {
  assert(WARP_SIZE == blockDim.x);
  assert(BLOCK_WARP == blockDim.y);
  size_t index = TILE_SIZE * blockIdx.x + threadIdx.y;
  const size_t last_index = min(TILE_SIZE * (blockIdx.x + 1), num_input);

  size_t i =  blockIdx.x * blockDim.x * blockDim.y + threadIdx.x * blockDim.y + threadIdx.y;
  // i is out of bound in num_random_states, so use a new curand
  curandState local_state;
  curand_init(i, 0, 0, &local_state);

  while (index < last_index) {
    const IdType rid = input[index];
    const IdType off = indptr[rid];
    const IdType len = indptr[rid + 1] - indptr[rid];

    if (len <= fanout) {
      size_t j = threadIdx.x;
      for (; j < len; j += WARP_SIZE) {
        tmp_src[index * fanout + j] = rid;
        tmp_dst[index * fanout + j] = indices[off + j];
      }
      __syncwarp();
      for (; j < fanout; j += WARP_SIZE) {
        tmp_src[index * fanout + j] = Constant::kEmptyKey;
        tmp_dst[index * fanout + j] = Constant::kEmptyKey;
      }
    } else {
      size_t j = threadIdx.x;
      for (; j < fanout; j += WARP_SIZE) {
        tmp_src[index * fanout + j] = rid;
        tmp_dst[index * fanout + j] = indices[off + j];
      }
      __syncwarp();
      for (; j < len; j += WARP_SIZE) {
        size_t k = curand(&local_state) % (j + 1);
        if (k < fanout) {
          atomicExch(tmp_dst + index * fanout + k, indices[off + j]);
        }
      }
    }
    index += BLOCK_WARP;
  }
}

#endif

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void partition_check_sample_khop0(
  const IdType *indptr, const IdType *indices,
  const IdType *input, const size_t num_input,
  const size_t fanout, IdType *tmp_src,
  IdType *tmp_dst, curandState *random_states,
  size_t num_random_states
) {
  assert(BLOCK_SIZE == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  curandState local_state;

  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    curand_init(2022, 0, 0, &local_state);  
    if (index < num_input) {
      const IdType rid = input[index];
      const IdType off = indptr[rid];
      const IdType len = indptr[rid + 1] - indptr[rid];
      if (len <= fanout) {
        size_t j = 0;
        for (; j < len; ++j) {
          tmp_src[index * fanout + j] = rid;
          tmp_dst[index * fanout + j] = indices[off + j];
        }

        for (; j < fanout; ++j) {
          tmp_src[index * fanout + j] = Constant::kEmptyKey;
          tmp_dst[index * fanout + j] = Constant::kEmptyKey;
        }
      } else {
        for (size_t j = 0; j < fanout; ++j) {
          tmp_src[index * fanout + j] = rid;
          tmp_dst[index * fanout + j] = indices[off + j];
        }

        for (size_t j = fanout; j < len; ++j) {
          size_t k = curand(&local_state) % (j + 1);
          if (k < fanout) {
            tmp_dst[index * fanout + k] = indices[off + j];
          }
        }
      }
    }
  }
}

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void count_edge(IdType *edge_src, size_t *item_prefix,
                           const size_t num_input, const size_t fanout) {
  assert(BLOCK_SIZE == blockDim.x);

  using BlockReduce = typename cub::BlockReduce<size_t, BLOCK_SIZE>;

  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  size_t count = 0;
#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_input) {
      for (size_t j = 0; j < fanout; j++) {
        if (edge_src[index * fanout + j] != Constant::kEmptyKey) {
          ++count;
        }
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
__global__ void compact_edge(const IdType *tmp_src, const IdType *tmp_dst,
                             IdType *out_src, IdType *out_dst, size_t *num_out,
                             const size_t *item_prefix, const size_t num_input,
                             const size_t fanout) {
  assert(BLOCK_SIZE == blockDim.x);

  using BlockScan = typename cub::BlockScan<size_t, BLOCK_SIZE>;

  constexpr const size_t VALS_PER_THREAD = TILE_SIZE / BLOCK_SIZE;

  __shared__ typename BlockScan::TempStorage temp_space;

  const size_t offset = item_prefix[blockIdx.x];

  BlockPrefixCallbackOp<size_t> prefix_op(0);

  // count successful placements
  for (size_t i = 0; i < VALS_PER_THREAD; ++i) {
    const size_t index = threadIdx.x + i * BLOCK_SIZE + blockIdx.x * TILE_SIZE;

    size_t item_per_thread = 0;
    if (index < num_input) {
      for (size_t j = 0; j < fanout; j++) {
        if (tmp_src[index * fanout + j] != Constant::kEmptyKey) {
          item_per_thread++;
        }
      }
    }

    size_t item_prefix_per_thread = item_per_thread;
    BlockScan(temp_space)
        .ExclusiveSum(item_prefix_per_thread, item_prefix_per_thread,
                      prefix_op);
    __syncthreads();

    for (size_t j = 0; j < item_per_thread; j++) {
      out_src[offset + item_prefix_per_thread + j] =
          tmp_src[index * fanout + j];
      out_dst[offset + item_prefix_per_thread + j] =
          tmp_dst[index * fanout + j];
    }
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *num_out = item_prefix[gridDim.x];
  }
}

}  // namespace

void GPUSampleKHop0(const IdType *indptr, const IdType *indices,
                    const IdType *input, const size_t num_input,
                    const size_t fanout, IdType *out_src, IdType *out_dst,
                    size_t *num_out, Context ctx, StreamHandle stream,
                    GPURandomStates *random_states, uint64_t task_key) {
  LOG(DEBUG) << "GPUSample: begin with num_input " << num_input
             << " and fanout " << fanout;
  Timer t0;
  const size_t num_tiles = RoundUpDiv(num_input, Constant::kCudaTileSize);
  const dim3 grid(num_tiles);
  const dim3 block(Constant::kCudaBlockSize);

  auto sampler_device = Device::Get(ctx);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  IdType *tmp_src = static_cast<IdType *>(
      sampler_device->AllocWorkspace(ctx, sizeof(IdType) * num_input * fanout));
  IdType *tmp_dst = static_cast<IdType *>(
      sampler_device->AllocWorkspace(ctx, sizeof(IdType) * num_input * fanout));
  LOG(DEBUG) << "GPUSample: cuda tmp_src malloc "
             << ToReadableSize(num_input * fanout * sizeof(IdType));
  LOG(DEBUG) << "GPUSample: cuda tmp_dst malloc "
             << ToReadableSize(num_input * fanout * sizeof(IdType));

  if(!RunConfig::partition_check) {
#ifndef NEW_ALGO
    sample_khop0<Constant::kCudaBlockSize, Constant::kCudaTileSize>
        <<<grid, block, 0, cu_stream>>>(
            indptr, indices, input, num_input, fanout, tmp_src, tmp_dst,
            random_states->GetStates(), random_states->NumStates());
    sampler_device->StreamSync(ctx, stream);
#else
    const int WARP_SIZE = 32;
    const int BLOCK_WARP = 128 / WARP_SIZE;
    const int TILE_SIZE = BLOCK_WARP * 16;
    const dim3 block_t(WARP_SIZE, BLOCK_WARP);
    const dim3 grid_t((num_input + TILE_SIZE - 1) / TILE_SIZE);
    sample_khop0<WARP_SIZE, BLOCK_WARP, TILE_SIZE> <<<grid_t, block_t, 0, cu_stream>>> (
            indptr, indices, input, num_input, fanout, tmp_src, tmp_dst,
            random_states->GetStates(), random_states->NumStates());
    sampler_device->StreamSync(ctx, stream);
#endif
  } else {
    const size_t num_tiles = (num_input + Constant::kCudaTileSize - 1) / Constant::kCudaTileSize;
    const dim3 gird(num_tiles);
    const dim3 block(Constant::kCudaBlockSize);
    partition_check_sample_khop0<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(
        indptr, indices, input, num_input, fanout, tmp_src, tmp_dst, 
        random_states->GetStates(), random_states->NumStates());
    sampler_device->StreamSync(ctx, stream);
  }
  double sample_time = t0.Passed();

  Timer t1;
  size_t *item_prefix = static_cast<size_t *>(
      sampler_device->AllocWorkspace(ctx, sizeof(size_t) * 2 * (grid.x + 1)));
  size_t *const item_prefix_out = &item_prefix[grid.x + 1];
  LOG(DEBUG) << "GPUSample: cuda item_prefix malloc "
             << ToReadableSize(sizeof(size_t) * 2 * (grid.x + 1));

  count_edge<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(tmp_src, item_prefix, num_input, fanout);
  sampler_device->StreamSync(ctx, stream);
  double count_edge_time = t1.Passed();

  Timer t2;
  size_t workspace_bytes;
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      nullptr, workspace_bytes, static_cast<size_t *>(nullptr),
      static_cast<size_t *>(nullptr), grid.x + 1, cu_stream));
  sampler_device->StreamSync(ctx, stream);

  void *workspace = sampler_device->AllocWorkspace(ctx, workspace_bytes);
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(workspace, workspace_bytes,
                                          item_prefix, item_prefix_out, grid.x + 1,
                                          cu_stream));
  sampler_device->StreamSync(ctx, stream);
  LOG(DEBUG) << "GPUSample: cuda workspace malloc "
             << ToReadableSize(workspace_bytes);
  if(RunConfig::partition_check) {
    size_t total_edge;
    sampler_device->CopyDataFromTo(item_prefix_out + grid.x, 0, &total_edge, 0, 
      sizeof(size_t), ctx, CPU());
    LOG(INFO) << __func__ << " total edge " << total_edge;
  }

  compact_edge<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(tmp_src, tmp_dst, out_src, out_dst,
                                      num_out, item_prefix_out, num_input, fanout);
  sampler_device->StreamSync(ctx, stream);
  double compact_edge_time = t2.Passed();

  sampler_device->FreeWorkspace(ctx, workspace);
  sampler_device->FreeWorkspace(ctx, item_prefix);
  sampler_device->FreeWorkspace(ctx, tmp_src);
  sampler_device->FreeWorkspace(ctx, tmp_dst);

  Profiler::Get().LogStepAdd(task_key, kLogL3KHopSampleCooTime, sample_time);
  Profiler::Get().LogStepAdd(task_key, kLogL3KHopSampleCountEdgeTime,
                             count_edge_time);
  Profiler::Get().LogStepAdd(task_key, kLogL3KHopSampleCompactEdgesTime,
                             compact_edge_time);

  LOG(DEBUG) << "GPUSample: succeed ";
}

template<size_t TILE_SIZE>
__global__ void count_partition_input(
  const IdType* input, const size_t num_input, const Id64Type* nodeId_map, 
  const size_t num_partition, IdType* tmp_partition_node_pos, IdType* partition_node_pos,
  IdType* partition_input_size
) {
  size_t block_start = blockIdx.x * TILE_SIZE;
  size_t block_end = block_start + TILE_SIZE;
  if(partition_node_pos == nullptr) {
    for(size_t i = block_start + threadIdx.x; i < block_end && i < num_input; i += blockDim.x) {
      IdType v = input[i];
      IdType p = reinterpret_cast<const IdType*>(&nodeId_map[v])[0];
      for(size_t j = 0; j < num_partition; j++) {
        tmp_partition_node_pos[i + j * num_input] = 0;
      }
      tmp_partition_node_pos[i + p * num_input] = 1;
    }
  } else {
    for(size_t i = block_start + threadIdx.x; i < block_end && i < num_input; i += blockDim.x) {
      IdType v = input[i];
      IdType p = reinterpret_cast<const IdType*>(&nodeId_map[v])[0];
      partition_node_pos[i] = tmp_partition_node_pos[i + p * num_input] - 1;
    }
    IdType idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < num_partition) {
      partition_input_size[idx] = tmp_partition_node_pos[(idx + 1) * num_input - 1];
    }
  }
}

template<size_t TILE_SIZE>
__global__ void create_partition_input(
  const IdType* input, const size_t num_input, 
  const Id64Type* nodeId_map, const IdType* partition_node_pos, 
  const IdType* partition_offset, IdType* partition_input, IdType* partition_node_pos_rmap
) {
  size_t block_start = blockIdx.x * TILE_SIZE;
  size_t block_end = block_start + TILE_SIZE;
  for(size_t i = block_start + threadIdx.x; i < block_end && i < num_input; i += blockDim.x) {
    IdType v = input[i];
    auto cur_nodeId_map = reinterpret_cast<const IdType*>(&nodeId_map[v]);
    IdType p = cur_nodeId_map[0];
    IdType id = cur_nodeId_map[1];
    IdType pos = partition_node_pos[i];
    IdType* cur_partition_input = partition_input + partition_offset[p];
    IdType* cur_partition_node_pos_rmap = partition_node_pos_rmap + partition_offset[p];
    cur_partition_input[pos] = id;
    cur_partition_node_pos_rmap[pos] = i;
  }
}

template<size_t TILE_SIZE>
__global__ void partition_sample_khop0(
  const IdType* indptr, const IdType* indices,
  const IdType* input, const size_t num_input, 
  const IdType* nodeId_rmap, const IdType* node_pos_map,
  const size_t fanout, IdType* tmp_src, IdType* tmp_dst,
  curandState *random_states, size_t num_random_states,
  bool partition_check = false
) {
  size_t block_start = blockIdx.x * TILE_SIZE;
  size_t blocke_end = block_start + TILE_SIZE;
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  curandState local_state;
  curand_init(i, 0, 0, &local_state);  

  for(size_t idx = block_start + threadIdx.x; idx < blocke_end && idx < num_input; idx += blockDim.x) {
    if(partition_check) {
      curand_init(2022, 0, 0, &local_state);
    }
    IdType rid = input[idx];
    IdType off = indptr[rid];
    IdType len = indptr[rid + 1] - indptr[rid];
    IdType pos = node_pos_map[idx]; 
    if(len <= fanout) {
      size_t j = 0;
      for(; j < len; j++) {
        tmp_src[pos * fanout + j] = nodeId_rmap[rid];
        tmp_dst[pos * fanout + j] = indices[off + j];
      }
      for(; j < fanout; j++) {
        tmp_src[pos * fanout + j] = Constant::kEmptyKey;
        tmp_dst[pos * fanout + j] = Constant::kEmptyKey;
      }
    } else {
      for(size_t j = 0; j < fanout; j++) {
        tmp_src[pos * fanout + j] = nodeId_rmap[rid];
        tmp_dst[pos * fanout + j] = indices[off + j];
      }
      for(size_t j = fanout; j < len; j++) {
        size_t k = curand(&local_state) % (j + 1);
        if(k < fanout) {
          tmp_dst[pos * fanout + k] = indices[off + j];        
        }
      }
    }
  }
}

void GPUPartitionSampleKHop0(
  const DisjointPartition &partition,
  const IdType *input, const size_t num_input, const size_t fanout,
  IdType *out_src, IdType *out_dst, size_t *num_out,
  Context ctx, StreamHandle stream,
  GPURandomStates *random_states, uint64_t task_key
) {
  LOG(DEBUG) << __func__ << " with " << num_input << " input, fanout " << fanout;
  auto sample_device = Device::Get(ctx);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  IdType* tmp_src = static_cast<IdType*>(
    sample_device->AllocWorkspace(ctx, sizeof(IdType) * num_input * fanout));
  IdType* tmp_dst = static_cast<IdType*>(
    sample_device->AllocWorkspace(ctx, sizeof(IdType) * num_input * fanout));
  auto max_buf_size = partition.GetMaxPartitionSize();
  IdType *indptr_buf[2], *indices_buf[2];
  for(int i = 0; i < 2; i++) {
    indptr_buf[i] = static_cast<IdType*>(
      sample_device->AllocWorkspace(ctx, sizeof(IdType) * max_buf_size.first));
    indices_buf[i] = static_cast<IdType*>(
      sample_device->AllocWorkspace(ctx, sizeof(IdType) * max_buf_size.second));
  }  
  LOG(DEBUG) << __func__ << " GPU alloc graph buf 2 * (" 
             << ToReadableSize(max_buf_size.first * sizeof(IdType)) << ", "
             << ToReadableSize(max_buf_size.second * sizeof(IdType)) << ")";

  // before sampling, partition input
  Timer t0;
  size_t num_tiles = (num_input + Constant::kCudaTileSize - 1) / Constant::kCudaTileSize; 
  dim3 grid(num_tiles);
  dim3 block(Constant::kCudaBlockSize);
  IdType* tmp_partition_node_pos = static_cast<IdType*>(
    sample_device->AllocWorkspace(ctx, 2 * sizeof(IdType) * num_input * partition.Size()));
  LOG(DEBUG) << __func__ << " GPU alloc tmp_partition_node_pos "
             << ToReadableSize(2 * sizeof(IdType) * num_input * partition.Size());
  count_partition_input<Constant::kCudaTileSize><<<grid, block, 0, cu_stream>>>(input, num_input, 
    partition.GetNodeIdMap(), partition.Size(), tmp_partition_node_pos, nullptr, nullptr);

  void* d_tmp_storage = nullptr;
  size_t tmp_storage_size = 0;
  CUDA_CALL(cub::DeviceScan::InclusiveSum(d_tmp_storage, tmp_storage_size, 
    tmp_partition_node_pos, tmp_partition_node_pos + num_input * partition.Size(),
    num_input, cu_stream));
  sample_device->StreamSync(ctx, cu_stream);
  d_tmp_storage = sample_device->AllocWorkspace(ctx, tmp_storage_size);
  for(int i = 0; i < partition.Size(); i++) {
    /// cub device scan
    size_t offset = i * num_input;
    CUDA_CALL(cub::DeviceScan::InclusiveSum(d_tmp_storage, tmp_storage_size, 
      tmp_partition_node_pos + offset, 
      tmp_partition_node_pos + offset + num_input * partition.Size(),
      num_input, cu_stream));
  }
  IdType* partition_node_pos = static_cast<IdType*>(
    sample_device->AllocWorkspace(ctx, sizeof(IdType) * num_input));
  IdType* d_partition_input_size = static_cast<IdType*>(
    sample_device->AllocWorkspace(ctx, sizeof(IdType) * partition.Size()));
  IdType* h_partition_input_size = static_cast<IdType*>(
    Device::Get(CPU())->AllocWorkspace(CPU(), sizeof(IdType) * partition.Size()));
  sample_device->StreamSync(ctx, cu_stream);
  count_partition_input<Constant::kCudaTileSize><<<grid, block, 0, cu_stream>>>(input, num_input,
    partition.GetNodeIdMap(), partition.Size(), tmp_partition_node_pos + num_input * partition.Size(), 
    partition_node_pos, d_partition_input_size);
  sample_device->CopyDataFromTo(
    d_partition_input_size, 0, h_partition_input_size, 0, 
    sizeof(IdType) * partition.Size(), ctx, CPU(), cu_stream);
  sample_device->StreamSync(ctx, cu_stream);
  LOG(DEBUG) << "partition input: " 
             << std::accumulate(h_partition_input_size, h_partition_input_size + partition.Size(), std::string{""}, 
                  [](const std::string& init, const IdType first) -> std::string {
                    return init + " " + std::to_string(first);
                });
  LOG(DEBUG) << "count partition input time " << t0.Passed();
  IdType h_partition_offset[partition.Size()];
  h_partition_offset[0] = 0;
  for(int i = 1; i < partition.Size(); i++) {
    h_partition_offset[i] = h_partition_offset[i-1] + h_partition_input_size[i-1];
  }
  IdType* d_partition_offset = static_cast<IdType*>(
    sample_device->AllocWorkspace(ctx, sizeof(IdType) * partition.Size()));
  sample_device->CopyDataFromTo(h_partition_offset, 0, d_partition_offset, 0,
    sizeof(IdType) * partition.Size(), CPU(), ctx);

  IdType* partition_input = static_cast<IdType*>(
    sample_device->AllocWorkspace(ctx, sizeof(IdType) * num_input));
  IdType* partition_node_pos_rmap = static_cast<IdType*>(
    sample_device->AllocWorkspace(ctx, sizeof(IdType) * num_input));
  create_partition_input<Constant::kCudaTileSize><<<grid, block, 0, cu_stream>>>(
    input, num_input, partition.GetNodeIdMap(), partition_node_pos, 
    d_partition_offset, partition_input, partition_node_pos_rmap);
  sample_device->StreamSync(ctx, cu_stream);
  LOG(DEBUG) << "create partition input done";
  LOG(DEBUG) << "create partition input time " << t0.Passed();
#if 0
  // check 
  IdType tmp_pos[num_input];
  sample_device->CopyDataFromTo(partition_node_pos_rmap, 0, tmp_pos, 0, 
    sizeof(IdType) * num_input, ctx, CPU());
  IdType pos_check[num_input] = {0};
  for(int i = 0; i < num_input; i++) {
    IdType pos = tmp_pos[i];
    if(pos_check[pos] != 0) {
      LOG(FATAL) << i << " bad pos " << pos;
    }
    pos_check[pos] = 1;
  }
  LOG(DEBUG) << "check position mapping done";
#endif
  auto Load = [&](IdType partitionId, IdType* indptr, IdType* indices, cudaStream_t load_stream) {
    auto &dataset = partition.Get(partitionId);
    cudaMemcpyAsync(indptr, dataset.indptr->Data(), dataset.indptr->NumBytes(), cudaMemcpyHostToDevice, load_stream);
    cudaMemcpyAsync(indices, dataset.indices->Data(), dataset.indices->NumBytes(), cudaMemcpyHostToDevice, load_stream);
  };
  
  Timer t1;
  cudaStream_t load_streams[partition.Size()];
  cudaStream_t sample_streams[partition.Size() + 1];
  for(IdType i = 0; i < partition.Size(); i++) {
    load_streams[i] = static_cast<cudaStream_t>(sample_device->CreateStream(ctx));
    sample_streams[i + 1] = static_cast<cudaStream_t>(sample_device->CreateStream(ctx));
  }
  LOG(DEBUG) << "create stream time " << t1.Passed();
  sample_device->StreamSync(ctx, cu_stream);
  IdType loadId = 0, sampleId = 0;
  LOG(DEBUG) << "start partition sampling ...";
  Timer t2;
  double partition_load_time = 0;
  double partition_sample_time = 0;
#if 0
  for(IdType i = 0; i < partition.Size() + 1; i++) {
    LOG(DEBUG) << "pipeline: " << i;
    if(i < partition.Size()) {
      if(i >= 2) {
        Timer t;
        sample_device->StreamSync(ctx, static_cast<StreamHandle>(sample_streams[i-1]));
        partition_sample_time += t.Passed();
      }
      LOG(DEBUG) << "|--loadId " << loadId;
      Load(i, indptr_buf[loadId], indices_buf[loadId], load_streams[i]);
      loadId = (loadId + 1) % 2;
    }
    if(i >= 1 && i < partition.Size() + 1) {
      Timer t;
      sample_device->StreamSync(ctx, static_cast<StreamHandle>(load_streams[i-1]));
      partition_load_time += t.Passed();
      IdType p = i - 1;
      IdType offset = h_partition_offset[p];
      size_t num_input = h_partition_input_size[p];
      size_t num_tiles = (num_input + Constant::kCudaTileSize - 1) / Constant::kCudaTileSize;
      const dim3 grid(num_tiles);
      const dim3 block(Constant::kCudaBlockSize);
      LOG(DEBUG) << "|--sampleId " << sampleId << " num_input " << num_input;
      partition_sample_khop0<Constant::kCudaTileSize><<<grid, block, 0, sample_streams[i]>>>(
        indptr_buf[sampleId], indices_buf[sampleId],
        partition_input + offset, num_input, 
        partition.GetNodeIdRMap(p), partition_node_pos_rmap + offset, 
        fanout, tmp_src, tmp_dst, random_states->GetStates(), random_states->NumStates(), 
        RunConfig::partition_check);
      sampleId = (sampleId + 1) % 2;
    } 
  }
  Timer t;
  sample_device->StreamSync(ctx, sample_streams[partition.Size()]);
  partition_sample_time += t.Passed();
#else 
  for(IdType i = 0; i < partition.Size(); i++) {
    Timer t0;
    Load(i, indptr_buf[0], indices_buf[0], cu_stream);
    sample_device->StreamSync(ctx, cu_stream);
    partition_load_time += t0.Passed();
    IdType offset = h_partition_offset[i];
    size_t num_input = h_partition_input_size[i];
    size_t num_tiles = (num_input + Constant::kCudaTileSize + 1) / Constant::kCudaTileSize;
    const dim3 grid(num_tiles);
    const dim3 block(Constant::kCudaBlockSize);
    Timer t1;
    partition_sample_khop0<Constant::kCudaTileSize><<<grid, block, 0, cu_stream>>>(
      indptr_buf[0], indices_buf[0],
      partition_input + offset, num_input, 
      partition.GetNodeIdRMap(i), partition_node_pos_rmap + offset, 
      fanout, tmp_src, tmp_dst, random_states->GetStates(), random_states->NumStates(), 
      RunConfig::partition_check);
    sample_device->StreamSync(ctx, cu_stream);
    partition_sample_time += t1.Passed();
  }
  sample_device->StreamSync(ctx, cu_stream);
#endif
  LOG(DEBUG) << "sampling done, start compact ...";
  double sample_time = t2.Passed();
  LOG(DEBUG) << "sampling time " << sample_time;
  // compact
  Timer t3;
  size_t* item_prefix = static_cast<size_t*>(
    sample_device->AllocWorkspace(ctx, sizeof(size_t) * 2 * (grid.x + 1)));
  size_t* item_prefix_out = &item_prefix[grid.x + 1];
  count_edge<Constant::kCudaBlockSize, Constant::kCudaTileSize>
    <<<grid, block, 0, cu_stream>>>(tmp_src, item_prefix, num_input, fanout);
  sample_device->StreamSync(ctx, stream);
  
  size_t workspace_bytes;
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
    nullptr, workspace_bytes, static_cast<size_t*>(nullptr),
    static_cast<size_t*>(nullptr), grid.x + 1, cu_stream));
  sample_device->StreamSync(ctx, cu_stream);

  void* workspace = sample_device->AllocWorkspace(ctx, workspace_bytes);
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
    workspace, workspace_bytes, 
    item_prefix, item_prefix_out,
    grid.x + 1, cu_stream));
  sample_device->StreamSync(ctx, cu_stream);
  if(RunConfig::partition_check) {
    size_t total_edge;
    sample_device->CopyDataFromTo(item_prefix_out + grid.x, 0, &total_edge, 0, 
      sizeof(size_t), ctx, CPU());
    LOG(INFO) << __func__ << " total edge " << total_edge;
  }

  compact_edge<Constant::kCudaBlockSize, Constant::kCudaTileSize>
    <<<grid, block, 0, cu_stream>>>(
      tmp_src, tmp_dst, out_src, out_dst, num_out,
      item_prefix_out, num_input, fanout);
  sample_device->StreamSync(ctx, stream);
  LOG(DEBUG) << "compact edge time " << t3.Passed();

  // free workspace
  Timer t4;
  Device::Get(CPU())->FreeWorkspace(CPU(), h_partition_input_size);
  sample_device->FreeWorkspace(ctx, d_tmp_storage);
  sample_device->FreeWorkspace(ctx, item_prefix);
  sample_device->FreeWorkspace(ctx, workspace);
  for(auto ts : {
    tmp_src, tmp_dst, 
    indptr_buf[0], indptr_buf[1], 
    indices_buf[0], indices_buf[1],
    tmp_partition_node_pos, partition_node_pos,
    partition_input, partition_node_pos_rmap,
    d_partition_input_size, d_partition_offset }
  ) {
    sample_device->FreeWorkspace(ctx, ts);
  }
  for(int i = 0; i < partition.Size(); i++) {
    sample_device->FreeStream(ctx, load_streams[i]);
    sample_device->FreeStream(ctx, sample_streams[i+1]);
  }
  LOG(INFO) << "free workspace time " << t4.Passed();
  LOG(INFO) << __func__ << " time " << t0.Passed();

  LOG(INFO) << "load_time " << partition_load_time << " sample_time " << partition_sample_time;
  Profiler::Get().LogStepAdd(task_key, kLogL3KHopSampleCooTime, sample_time);
  Profiler::Get().LogStepAdd(task_key, kLogL3KHopPartitionSampleLoadTime, partition_load_time);
  Profiler::Get().LogStepAdd(task_key, kLogL3KHopPartitionSampleTime, partition_sample_time);
}

}  // namespace cuda
}  // namespace common
}  // namespace samgraph
