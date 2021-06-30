#include <curand.h>
#include <curand_kernel.h>

#include <algorithm>
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

namespace samgraph {
namespace common {
namespace cuda {

namespace {

__global__ void sample_khop1(const IdType *indptr, const IdType *indices,
                             const IdType *input, const size_t num_input,
                             const size_t fanout, IdType *tmp_src,
                             IdType *tmp_dst, curandState *random_states,
                             size_t num_random_states) {
  size_t num_task = num_input * fanout;
  size_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
  size_t task_span = blockDim.x * gridDim.x;

  assert(threadId < num_random_states);
  // cache the curand state
  curandState local_state = random_states[threadId];

  for (size_t task_idx = threadId; task_idx < num_task; task_idx += task_span) {
    const IdType rid = input[task_idx / fanout];
    const IdType off = indptr[rid];
    const IdType len = indptr[rid + 1] - indptr[rid];

    if (len == 0) {
      tmp_src[task_idx] = Constant::kEmptyKey;
      tmp_dst[task_idx] = Constant::kEmptyKey;
    } else {
      size_t k = curand(&local_state) % len;
      tmp_src[task_idx] = rid;
      tmp_dst[task_idx] = indices[off + k];
    }
  }
  // restore the state
  random_states[threadId] = local_state;
}

__global__ void count_edge(IdType *src, IdType *dst, size_t *item_prefix,
                           size_t num_task) {
  size_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
  size_t task_span = blockDim.x * gridDim.x;

  for (size_t task_idx = threadId; task_idx < num_task; task_idx += task_span) {
    if (task_idx < (num_task - 1)) {
      // when the thread is the last thread to get the value, it should be
      // the one responsible for copying the edge to the output array
      item_prefix[task_idx] = (src[task_idx] != src[task_idx + 1] ||
                               dst[task_idx] != dst[task_idx + 1]) &&
                              src[task_idx] != Constant::kEmptyKey;
    } else {
      item_prefix[task_idx] = src[task_idx] != Constant::kEmptyKey;
    }
  }

  if (threadId == 0) {
    item_prefix[num_task] = 0;
  }
}

__global__ void compact_edge(IdType *tmp_src, IdType *tmp_dst, IdType *out_src,
                             IdType *out_dst, size_t *item_prefix,
                             size_t num_task, size_t *num_out) {
  size_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
  size_t task_span = blockDim.x * gridDim.x;

  for (size_t task_idx = threadId; task_idx < num_task; task_idx += task_span) {
    bool cond;
    if (task_idx < (num_task - 1)) {
      cond = (tmp_src[task_idx] != tmp_src[task_idx + 1] ||
              tmp_dst[task_idx] != tmp_dst[task_idx + 1]) &&
             tmp_src[task_idx] != Constant::kEmptyKey;
    } else {
      cond = tmp_src[task_idx] != Constant::kEmptyKey;
    }

    if (cond) {
      out_src[item_prefix[task_idx]] = tmp_src[task_idx];
      out_dst[item_prefix[task_idx]] = tmp_dst[task_idx];
    }

    // out_src[item_prefix[task_idx]] = tmp_src[task_idx];
    // out_dst[item_prefix[task_idx]] = tmp_dst[task_idx];
  }

  if (threadId == 0) {
    *num_out = item_prefix[num_task];
  }
}

}  // namespace

void GPUSampleKHop1(const IdType *indptr, const IdType *indices,
                    const IdType *input, const size_t num_input,
                    const size_t fanout, IdType *out_src, IdType *out_dst,
                    size_t *num_out, Context ctx, StreamHandle stream,
                    GPURandomStates *random_states, uint64_t task_key) {
  LOG(DEBUG) << "GPUSample: begin with num_input " << num_input
             << " and fanout " << fanout;
  Timer t0;

  auto sampler_device = Device::Get(ctx);
  auto cu_stream = static_cast<cudaStream_t>(stream);
  auto num_sample = num_input * fanout;

  IdType *tmp_src = static_cast<IdType *>(
      sampler_device->AllocWorkspace(ctx, sizeof(IdType) * num_sample));
  IdType *tmp_dst = static_cast<IdType *>(
      sampler_device->AllocWorkspace(ctx, sizeof(IdType) * num_sample));
  LOG(DEBUG) << "GPUSample: cuda tmp_src malloc "
             << ToReadableSize(num_sample * sizeof(IdType));
  LOG(DEBUG) << "GPUSample: cuda tmp_dst malloc "
             << ToReadableSize(num_sample * sizeof(IdType));

  // 1. Sampling
  size_t num_threads = Min(num_sample, Constant::kKHop1MaxThreads);
  const dim3 grid(
      RoundUpDiv(num_threads, static_cast<size_t>(Constant::kCudaBlockSize)));
  const dim3 block(Constant::kCudaBlockSize);
  sample_khop1<<<grid, block, 0, cu_stream>>>(
      indptr, indices, input, num_input, fanout, tmp_src, tmp_dst,
      random_states->GetStates(), random_states->NumStates());
  sampler_device->StreamSync(ctx, stream);

  double sample_time = t0.Passed();
  LOG(DEBUG) << "GPUSample: kernel sampling, time cost: " << sample_time;

  // 2. Remove duplication.
  // 2.1 COO pair sorting
  Timer t1;
  size_t temp_storage_bytes = 0;
  CUDA_CALL(cub::DeviceRadixSort::SortPairs(
      nullptr, temp_storage_bytes, tmp_src, tmp_src, tmp_dst, tmp_dst,
      num_sample, 0, sizeof(IdType) * 8, cu_stream));
  sampler_device->StreamSync(ctx, stream);

  void *d_temp_storage =
      sampler_device->AllocWorkspace(ctx, temp_storage_bytes);
  CUDA_CALL(cub::DeviceRadixSort::SortPairs(
      d_temp_storage, temp_storage_bytes, tmp_src, tmp_src, tmp_dst, tmp_dst,
      num_sample, 0, sizeof(IdType) * 8, cu_stream));
  sampler_device->StreamSync(ctx, stream);
  sampler_device->FreeWorkspace(ctx, d_temp_storage);
  double sort_coo_time = t1.Passed();
  LOG(DEBUG) << "GPUSample: sort the temporary results, time cost: "
             << sort_coo_time;

  // 2.2 Count edges
  Timer t2;
  size_t *item_prefix = static_cast<size_t *>(
      sampler_device->AllocWorkspace(ctx, sizeof(size_t) * num_sample + 1));
  LOG(DEBUG) << "GPUSample: cuda prefix_num malloc "
             << ToReadableSize(sizeof(int) * num_sample);
  count_edge<<<grid, block, 0, cu_stream>>>(tmp_src, tmp_dst, item_prefix,
                                            num_sample);
  sampler_device->StreamSync(ctx, stream);

  temp_storage_bytes = 0;
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes,
                                          item_prefix, item_prefix,
                                          num_sample + 1, cu_stream));
  sampler_device->StreamSync(ctx, stream);

  d_temp_storage = sampler_device->AllocWorkspace(ctx, temp_storage_bytes);
  LOG(DEBUG) << "GPUSample: cuda temp_storage for ExclusiveSum malloc "
             << ToReadableSize(temp_storage_bytes);
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                          item_prefix, item_prefix,
                                          num_sample + 1, cu_stream));
  sampler_device->StreamSync(ctx, stream);
  sampler_device->FreeWorkspace(ctx, d_temp_storage);
  double prefix_sum_time = t2.Passed();
  LOG(DEBUG) << "GPUSample: ExclusiveSum time cost: " << prefix_sum_time;

  // 2.3 Compact edges
  Timer t3;
  compact_edge<<<grid, block, 0, cu_stream>>>(
      tmp_src, tmp_dst, out_src, out_dst, item_prefix, num_sample, num_out);
  sampler_device->StreamSync(ctx, stream);
  double compact_edge_time = t3.Passed();
  LOG(DEBUG) << "GPUSample: compact_edge time cost: " << compact_edge_time;

  sampler_device->FreeWorkspace(ctx, item_prefix);
  sampler_device->FreeWorkspace(ctx, tmp_src);
  sampler_device->FreeWorkspace(ctx, tmp_dst);

  Profiler::Get().LogStepAdd(task_key, kLogL3SampleCooTime, sample_time);
  Profiler::Get().LogStepAdd(task_key, kLogL3SampleSortCooTime, sort_coo_time);
  Profiler::Get().LogStepAdd(task_key, kLogL3SampleCountEdgeTime,
                             prefix_sum_time);
  Profiler::Get().LogStepAdd(task_key, kLogL3SampleCompactEdgesTime,
                             compact_edge_time);

  double total_time = t0.Passed();
  LOG(DEBUG) << "GPUSample: succeed total time cost: " << total_time;
}

}  // namespace cuda
}  // namespace common
}  // namespace samgraph
