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
#include "../run_config.h"
#include "cuda_function.h"
#include "cuda_utils.h"
#include "cuda_engine.h"

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
    Timer _kt;

    {
      sampler_device->StreamSync(ctx, stream);
      IdType *p2p_indptr, *p2p_indice;
      IdType *maped_indptr, *mapped_indices;  
      IdType num_node = GPUEngine::Get()->GetGraphDataset()->num_node;
      IdType num_edge = GPUEngine::Get()->GetGraphDataset()->num_edge;
      CUDA_CALL(cudaHostAlloc(&maped_indptr, sizeof(IdType) * (GPUEngine::Get()->GetGraphDataset()->num_node + 1), cudaHostAllocMapped));
      CUDA_CALL(cudaHostAlloc(&mapped_indices, sizeof(IdType) * (GPUEngine::Get()->GetGraphDataset()->num_edge), cudaHostAllocMapped));
      CUDA_CALL(cudaSetDevice(2));
      CUDA_CALL(cudaMalloc(&p2p_indptr, sizeof(IdType) * (GPUEngine::Get()->GetGraphDataset()->num_node + 1)));
      CUDA_CALL(cudaMalloc(&p2p_indice, sizeof(IdType) * (GPUEngine::Get()->GetGraphDataset()->num_edge)));
      CUDA_CALL(cudaSetDevice(1));
      CUDA_CALL(cudaDeviceEnablePeerAccess(2, 0));

      CUDA_CALL(cudaMemcpy(maped_indptr, indptr, sizeof(IdType) * (num_node + 1), cudaMemcpyDefault));
      CUDA_CALL(cudaMemcpy(mapped_indices, indices, sizeof(IdType) * (num_edge), cudaMemcpyDefault));
      CUDA_CALL(cudaMemcpy(p2p_indptr, indptr, sizeof(IdType) * (num_node + 1), cudaMemcpyDefault));
      CUDA_CALL(cudaMemcpy(p2p_indice, indices, sizeof(IdType) * (num_edge), cudaMemcpyDefault));

      LOG(INFO) << "sample_khop0 input=" << num_input << " fanout=" << fanout; 
      Timer p2p_tm;
      sample_khop0<WARP_SIZE, BLOCK_WARP, TILE_SIZE> <<<grid_t, block_t, 0, cu_stream>>> (
            p2p_indptr, p2p_indice, input, num_input, fanout, tmp_src, tmp_dst,
            random_states->GetStates(), random_states->NumStates());
      sampler_device->StreamSync(ctx, stream);
      LOG(INFO) << "p2p " << p2p_tm.Passed();
      Timer maped_tm;
      sample_khop0<WARP_SIZE, BLOCK_WARP, TILE_SIZE> <<<grid_t, block_t, 0, cu_stream>>> (
            maped_indptr, mapped_indices, input, num_input, fanout, tmp_src, tmp_dst,
            random_states->GetStates(), random_states->NumStates());
      sampler_device->StreamSync(ctx, stream);
      LOG(INFO) << "mappded " << maped_tm.Passed();
      CHECK(false);
    }

    sample_khop0<WARP_SIZE, BLOCK_WARP, TILE_SIZE> <<<grid_t, block_t, 0, cu_stream>>> (
            indptr, indices, input, num_input, fanout, tmp_src, tmp_dst,
            random_states->GetStates(), random_states->NumStates());
    sampler_device->StreamSync(ctx, stream);
    LOG(INFO) << "sample_khop0 input=" << num_input 
              << " fanout=" << fanout << " " << _kt.Passed() << "s";
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

}  // namespace cuda
}  // namespace common
}  // namespace samgraph
