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
#include "cuda_function.h"
#include "cuda_utils.h"

#define NEW_ALGO

namespace samgraph {
namespace common {
namespace cuda {

namespace {
#ifndef NEW_ALGO
template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void sample_khop2(const IdType *indptr, IdType *indices,
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
          // now we have $len - j$ cancidate
          size_t selected_j = curand(&local_state) % (len - j);
          IdType selected_node_id = indices[off + selected_j];
          tmp_src[index * fanout + j] = rid;
          tmp_dst[index * fanout + j] = selected_node_id;
          // swap indices[off+len-j-1] with indices[off+ selected_j];
          indices[off + selected_j] = indices[off+len-j-1];
          indices[off+len-j-1] = selected_node_id;
        }
      }
    }
  }

  random_states[i] = local_state;
}
#else

constexpr int HASH_SHIFT = 7;
constexpr int HASH_MASK  = ((1 << HASH_SHIFT) - 1);
constexpr int HASHTABLE_SIZE = (1 << HASH_SHIFT);

struct HashItem {
  IdType key_val;
};

__device__ IdType _HashtableInsertVersion(HashItem* hashtable, IdType key, IdType version) {
  IdType pos = (key & HASH_MASK);
  IdType offset = 1;
  while (true) {
    IdType old = atomicCAS(&hashtable[pos].key_val, 0xffffffff, key);
    if (old == 0xffffffff || old == key) {
      return pos;
    }
    pos = ((pos + offset) & HASH_MASK);
    offset += 1;
  }
}

__device__ IdType _HashtableSwapAt(HashItem* hashtable, IdType pos, IdType val) {
  return atomicExch(&hashtable[pos].key_val, val);
}

template <size_t WARP_SIZE, size_t BLOCK_WARP, size_t TILE_SIZE>
__global__ void sample_khop2(const IdType *indptr, IdType *indices,
                             const IdType *input, const size_t num_input,
                             const size_t fanout, IdType *tmp_src,
                             IdType *tmp_dst, curandState *random_states,
                             size_t num_random_states) {
  assert(WARP_SIZE == blockDim.x);
  assert(BLOCK_WARP == blockDim.y);
  IdType index = TILE_SIZE * blockIdx.x + threadIdx.y;
  const IdType last_index = min(TILE_SIZE * (blockIdx.x + 1), num_input);

  IdType out_row = blockIdx.x * TILE_SIZE + threadIdx.y;

  __shared__ HashItem shared_hashtable[BLOCK_WARP][HASHTABLE_SIZE];

  auto hashtable = shared_hashtable[threadIdx.y];
  for (int idx = threadIdx.x; idx < HASHTABLE_SIZE; idx += WARP_SIZE) {
    hashtable[idx].key_val = 0xffffffff;
    // hashtable[idx].version = 0xffffffff;
    // hashtable[idx].flag = 0;
  }

  IdType i =  blockIdx.x * blockDim.y + threadIdx.y;
  assert(i < num_random_states);
  curandState local_state = random_states[i];

  for (; index < TILE_SIZE * (blockIdx.x + 1); index += BLOCK_WARP) {
    if (index >= num_input) {
      __syncthreads();
      __syncthreads();
      __syncthreads();
      continue;
    }
    const IdType rid = input[index];
    const IdType off = indptr[rid];
    const IdType len = indptr[rid + 1] - indptr[rid];

    if (len <= fanout) {
      size_t j = threadIdx.x;
      for (; j < len; j += WARP_SIZE) {
        tmp_src[index * fanout + j] = rid;
        tmp_dst[index * fanout + j] = indices[off + j];
      }

      for (; j < fanout; j += WARP_SIZE) {
        tmp_src[index * fanout + j] = Constant::kEmptyKey;
        tmp_dst[index * fanout + j] = Constant::kEmptyKey;
      }
      __syncthreads();
      __syncthreads();
    } else {
      IdType* hash_pos = &tmp_src[index * fanout];
      for (size_t j = threadIdx.x; j < fanout; j += WARP_SIZE) {
        IdType rand = (curand(&local_state) % (len - j)) + j;
        hash_pos[j] = _HashtableInsertVersion(hashtable, rand, out_row);
      }
      __syncthreads();
      for (size_t j = threadIdx.x; j < fanout; j += WARP_SIZE) {
        IdType val = _HashtableSwapAt(hashtable, hash_pos[j], j);
        // tmp_src[index * fanout + j] = rid;
        tmp_dst[index * fanout + j] = indices[off + val];
      }
      __syncthreads();
      for (size_t j = threadIdx.x; j < fanout; j += WARP_SIZE) {
        hashtable[hash_pos[j]].key_val = 0xffffffff;
        tmp_src[index * fanout + j] = rid;
      }
    }
    __syncthreads();
  }
  random_states[i] = local_state;
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

void GPUSampleKHop2(const IdType *indptr, IdType *indices,
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
  sample_khop2<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(
          indptr, indices, input, num_input, fanout, tmp_src, tmp_dst,
          random_states->GetStates(), random_states->NumStates());
#else
  static_assert(sizeof(unsigned long long) == 8, "");
  const int WARP_SIZE = 32;
  const int BLOCK_WARP = 128 / WARP_SIZE;
  const int TILE_SIZE = BLOCK_WARP * 16;
  const dim3 block_t(WARP_SIZE, BLOCK_WARP);
  const dim3 grid_t((num_input + TILE_SIZE - 1) / TILE_SIZE);
  sample_khop2<WARP_SIZE, BLOCK_WARP, TILE_SIZE> <<<grid_t, block_t, 0, cu_stream>>> (
          indptr, indices, input, num_input, fanout, tmp_src, tmp_dst,
          random_states->GetStates(), random_states->NumStates());
#endif
  sampler_device->StreamSync(ctx, stream);
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