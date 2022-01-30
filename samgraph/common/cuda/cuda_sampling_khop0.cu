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
  const IdType* input, const size_t num_input, 
  const Id64Type* nodeId_map, IdType* partition_input_size
) {
  size_t block_start = blockIdx.x * TILE_SIZE;
  size_t block_end = block_start + TILE_SIZE;
  for(size_t i = block_start + threadIdx.x; i < block_end && i < num_input; i += blockDim.x) {
    auto v = input[i];
    auto cur_nodeId_map = reinterpret_cast<IdType*>(nodeId_map[v]);
    auto p = cur_nodeId_map[0];
    partition_input_size[p]++;
  }
}

template<size_t TILE_SIZE>
__global__ void create_partition_input(
  const IdType* input, const size_t num_input, 
  const Id64Type* nodeId_map, IdType** partition_input, IdType** partition_node_pos_map,
  IdType* partition_input_size, const size_t num_partition
) {
  auto i = blockDim.x * blockDim.x + threadIdx.x; 
  if(i < num_partition) {
    partition_input_size[i] = 0;
  } 
  size_t block_start = blockIdx.x * TILE_SIZE;
  size_t block_end = block_start + TILE_SIZE;
  for(size_t i = block_start + threadIdx.x; i < block_end && i < num_input; i += blockDim.x) {
    auto v = input[i];
    auto cur_nodeId_map = reinterpret_cast<IdType*>(nodeId_map[v]);
    auto p = cur_nodeId_map[0];
    auto id = cur_nodeId_map[1];
    partition_input[p][partition_input_size[p]] = id;
    partition_node_pos_map[p][partition_input_size[p]] = i;
    partition_input_size[p]++;
  }
}

template<size_t TILE_SIZE>
__global__ void partition_sample_khop0(
  const IdType* indptr, const IdType* indices,
  const IdType* input, const size_t num_input, 
  const IdType* nodeId_rmap, const IdType* node_pos_map,
  const size_t fanout, IdType* tmp_src, IdType* tmp_dst,
  curandState *random_states, size_t num_random_states
) {
  size_t block_start = blockIdx.x * TILE_SIZE;
  size_t blocke_end = block_start + TILE_SIZE;
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  curandState local_state;
  curand_init(i, 0, 0, &local_state);  

  for(size_t idx = block_start + threadIdx.x; i < blocke_end && i < num_input; i += blockDim.x) {
    IdType rid = input[idx];
    IdType off = indices[rid];
    IdType len = indptr[rid + 1] - indptr[rid];
    IdType pos_idx = node_pos_map[idx]; 
    if(len <= fanout) {
      size_t j = 0;
      for(; j < len; j++) {
        tmp_src[pos_idx * fanout + j] = nodeId_rmap[rid];
        tmp_dst[pos_idx * fanout + j] = indices[off + j];
      }
      for(; j < fanout; j++) {
        tmp_src[pos_idx * fanout + j] = Constant::kEmptyKey;
        tmp_dst[pos_idx * fanout + j] = Constant::kEmptyKey;
      }
    } else {
      for(size_t j = 0; j < fanout; j++) {
        tmp_src[pos_idx * fanout + j] = nodeId_rmap[rid];
        tmp_dst[pos_idx * fanout + j] = indices[off + j];
      }
      for(size_t j = fanout; j < len; j++) {
        size_t k = curand(&local_state) % (j + 1);
        if(k < fanout) {
          tmp_dst[pos_idx * fanout + k] = indices[off + j];        
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

  // before sampling, partition input
  IdType* d_partition_input_size = static_cast<IdType*>(
    sample_device->AllocWorkspace(ctx, sizeof(IdType) * partition.Size()));
  IdType* h_partition_input_size = static_cast<IdType*>(
    Device::Get(CPU())->AllocWorkspace(CPU(), sizeof(IdType) * partition.Size()));
  size_t num_tiles = (num_input + Constant::kCudaTileSize - 1) / Constant::kCudaTileSize; 
  dim3 grid(num_tiles);
  dim3 block(Constant::kCudaBlockSize);
  count_partition_input<Constant::kCudaTileSize><<<block, grid, 0, cu_stream>>>(
    input, num_input, partition.GetNodeIdMap(), d_partition_input_size);
  sample_device->CopyDataFromTo(
    d_partition_input_size, 0, h_partition_input_size, 0, 
    sizeof(IdType) * partition.Size(), ctx, CPU(), cu_stream);
  sample_device->StreamSync(ctx, cu_stream);

  IdType* partition_input[partition.Size()];
  IdType* partition_node_pos_map[partition.Size()];
  for(int i = 0; i < partition.Size(); i++) {
    partition_input[i] = static_cast<IdType*>(
      sample_device->AllocWorkspace(ctx, sizeof(IdType) * h_partition_input_size[i]));
    partition_node_pos_map[i] = static_cast<IdType*>(
      sample_device->AllocWorkspace(ctx, sizeof(IdType) * h_partition_input_size[i]));
  }
  create_partition_input<Constant::kCudaTileSize><<<block, grid, 0, cu_stream>>>(
    input, num_input, partition.GetNodeIdMap(), 
    partition_input, partition_node_pos_map,
    d_partition_input_size, partition.Size());
  
  auto Load = [&](IdType partitionId, IdType* indptr, IdType* indices, cudaStream_t load_stream) {
    auto &dataset = partition.Get(partitionId);
    cudaMemcpyAsync(indptr, dataset.indptr->Data(), dataset.indptr->NumBytes(), cudaMemcpyHostToDevice, load_stream);
    cudaMemcpyAsync(indices, dataset.indices->Data(), dataset.indices->NumBytes(), cudaMemcpyHostToDevice, load_stream);
  };
  
  cudaStream_t load_streams[partition.Size()];
  cudaStream_t sample_streams[partition.Size() + 1];
  for(IdType i = 0; i < partition.Size(); i++) {
    load_streams[i] = static_cast<cudaStream_t>(sample_device->CreateStream(ctx));
    sample_streams[i + 1] = static_cast<cudaStream_t>(sample_device->CreateStream(ctx));
  }
  sample_device->StreamSync(ctx, cu_stream);
  IdType loadId = 0, sampleId = 0;
  for(IdType i = 0; i < partition.Size() + 2; i++) {
    if(i < partition.Size()) {
      if(i >= 2) {
        sample_device->StreamSync(ctx, static_cast<StreamHandle>(sample_streams[i-1]));
      }
      Load(i, indptr_buf[loadId], indices_buf[loadId], load_streams[i]);
      loadId = (loadId + 1) % 2;
    }
    if(i >= 1 && i < partition.Size() + 1) {
      sample_device->StreamSync(ctx, static_cast<StreamHandle>(load_streams[i-1]));
      IdType p = i - 1;
      size_t num_input = h_partition_input_size[p];
      size_t num_tiles = (num_input + Constant::kCudaTileSize - 1) / Constant::kCudaTileSize;
      const dim3 grid(num_tiles);
      const dim3 block(Constant::kCudaBlockSize);
      partition_sample_khop0<Constant::kCudaTileSize><<<block, grid, 0, sample_streams[i]>>>(
        indptr_buf[sampleId], indices_buf[sampleId],
        partition_input[p], h_partition_input_size[sampleId], 
        partition.GetNodeIdRMap(p), partition_node_pos_map[p],
        fanout, tmp_src, tmp_dst, random_states->GetStates(), random_states->NumStates());
      sampleId = (sampleId + 1) % 2;
    } 
  }
  sample_device->StreamSync(ctx, sample_streams[partition.Size()]);

  // compact
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

  compact_edge<Constant::kCudaBlockSize, Constant::kCudaTileSize>
    <<<grid, block, 0, cu_stream>>>(
      tmp_src, tmp_dst, out_src, out_dst, num_out,
      item_prefix_out, num_input, fanout);
  sample_device->StreamSync(ctx, stream);

  // free workspace
  Device::Get(CPU())->FreeWorkspace(CPU(), h_partition_input_size);
  for(auto ts : {
    tmp_src, tmp_dst, 
    indptr_buf[0], indptr_buf[1], 
    indices_buf[0], indices_buf[1],
    d_partition_input_size }
  ) {
    sample_device->FreeWorkspace(ctx, ts);
  }
  for(int i = 0; i < partition.Size(); i++) {
    sample_device->FreeWorkspace(ctx, partition_input[i]);
    sample_device->FreeWorkspace(ctx, partition_node_pos_map[i]);
  }
  for(int i = 0; i < partition.Size(); i++) {
    sample_device->FreeStream(ctx, load_streams[i]);
    sample_device->FreeStream(ctx, sample_streams[i+1]);
  }
}

}  // namespace cuda
}  // namespace common
}  // namespace samgraph
