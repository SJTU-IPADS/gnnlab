#include <cassert>
#include <chrono>
#include <cstdio>

#include <cub/cub.cuh>
#include <curand.h>
#include <curand_kernel.h>

#include "../common.h"
#include "../constant.h"
#include "../device.h"
#include "../logging.h"
#include "../profiler.h"
#include "../timer.h"
#include "cuda_function.h"

namespace {

template <typename T> struct BlockPrefixCallbackOp {
  T _running_total;

  __device__ BlockPrefixCallbackOp(const T running_total)
      : _running_total(running_total) {}

  __device__ T operator()(const T block_aggregate) {
    const T old_prefix = _running_total;
    _running_total += block_aggregate;
    return old_prefix;
  }
};

} // namespace


namespace samgraph {
namespace common {
namespace cuda {

__global__ void nextSample(const IdType *indptr, const IdType *indices, 
                           const IdType *input, const size_t num_input, 
                           const size_t fanout, IdType *tmp_src,
                           IdType *tmp_dst) {
  size_t num_task = num_input * fanout;
  int threadId = threadIdx.x + blockDim.x * blockIdx.x;
  int max_span = blockDim.x * gridDim.x;
//  curandState state;
//  curand_init(num_input, threadId, 0, &state);

  for (int task_start = threadId; task_start < num_task; task_start += max_span) {
    assert((task_start / fanout) < num_input);
    const IdType rid = input[task_start / fanout];
    const IdType off = indptr[rid];
    const IdType len = indptr[rid + 1] - indptr[rid];
//    size_t k = curand(&state) % len;
    size_t k = (task_start % fanout);
    tmp_src[task_start] = rid;
    if (k < len) {
        tmp_dst[task_start] = indices[off + k];
    }
    else {
        tmp_src[task_start] = Constant::kEmptyKey;
        tmp_dst[task_start] = Constant::kEmptyKey;
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
      // printf("index %lu  count %lu\n", index, count);
    }
  }

  __shared__ typename BlockReduce::TempStorage temp_space;

  count = BlockReduce(temp_space).Sum(count);

  if (threadIdx.x == 0) {
    item_prefix[blockIdx.x] = count;
    // printf("blockIdx.x %d count %lu\n", blockIdx.x, count);
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
    // printf("item_prefix %d\n", item_prefix[gridDim.x]);
  }
}

/**
 * @brief sampling algorithm from nextdoor
 * CSR format example:
        ROW_INDEX = [  0  2  4  7  8  ]
        COL_INDEX = [  0  1  1  3  2  3  4  5  ]
        V         = [ 10 20 30 40 50 60 70 80  ]
 * @param indptr      ROW_INDEX, sampling vertices
 * @param indices     COL_INDEX, neighbors
 * @param input       the indices of sampling vertices
 * @param num_input   the number of sampling vertices
 * @param fanout      the number of neighbors for each sampling vertex
 * @param out_src     src vertices of all neighbors 
 * @param out_dst     dst vertices of all neighbors
 * @param num_out     the number of all neighbors
 * @param ctx         GPU contex
 * @param stream      GPU stream
 * @param task_key
 */
void GPUNextdoorSample(const IdType *indptr, const IdType *indices,
                       const IdType *input, const size_t num_input,
                       const size_t fanout, IdType *out_src, IdType *out_dst,
                       size_t *num_out, Context ctx, StreamHandle stream,
                       uint64_t task_key) {
  LOG(DEBUG) << "GPUSample: begin with num_input " << num_input
             << " and fanout " << fanout;
  Timer t0;

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

  const size_t max_threads = 5 * 1024 * 1024;
  size_t num_threads = num_input * fanout;
  if (num_threads > max_threads) {
      num_threads = max_threads;
  }
  const dim3 grid((num_threads + Constant::kCudaBlockSize - 1) / Constant::kCudaBlockSize);
  const dim3 block(Constant::kCudaBlockSize);
  /*
  CHECKPOINT(2, "grid: %d, %d, %d\n", 
          grid.x, grid.y, grid.z);
  CHECKPOINT(3, "block: %d, %d, %d\n", 
          block.x, block.y, block.z);
  CHECKPOINT(4, "num_input: %d\n", num_input);
  CHECKPOINT(5, "fanout: %d\n", fanout);
  CHECKPOINT(6, "num_threads: %d\n", num_threads);
  */
  // FIXME: add random module
  nextSample<<<grid, block, 0, cu_stream>>>(indptr, indices, input, num_input, fanout,
                                            tmp_src, tmp_dst);
  sampler_device->StreamSync(ctx, stream);

  double sample_time = t0.Passed();

  Timer t1;
  size_t *item_prefix = static_cast<size_t *>(
      sampler_device->AllocWorkspace(ctx, sizeof(size_t) * (grid.x + 1)));
  LOG(DEBUG) << "GPUSample: cuda item_prefix malloc "
             << ToReadableSize(sizeof(size_t) * (grid.x + 1));

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
                                          item_prefix, item_prefix, grid.x + 1,
                                          cu_stream));
  sampler_device->StreamSync(ctx, stream);
  LOG(DEBUG) << "GPUSample: cuda workspace malloc "
             << ToReadableSize(workspace_bytes);

  compact_edge<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(tmp_src, tmp_dst, out_src, out_dst,
                                      num_out, item_prefix, num_input, fanout);
  sampler_device->StreamSync(ctx, stream);
  double compact_edge_time = t2.Passed();

  sampler_device->FreeWorkspace(ctx, workspace);
  sampler_device->FreeWorkspace(ctx, item_prefix);
  sampler_device->FreeWorkspace(ctx, tmp_src);
  sampler_device->FreeWorkspace(ctx, tmp_dst);

  LOG(DEBUG) << "GPUSample: succeed ";
}

} // namespace cuda
} // namespace common
} // namespace samgraph

