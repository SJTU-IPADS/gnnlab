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


namespace samgraph {
namespace common {
namespace cuda {

__global__ void nextSample(const IdType *indptr, const IdType *indices, 
                           const IdType *input, const size_t num_input, 
                           const size_t fanout, IdType *tmp_src,
                           IdType *tmp_dst, curandState* states, size_t num_seeds) {
  size_t num_task = num_input * fanout;
  size_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
  size_t max_span = blockDim.x * gridDim.x;
  assert(threadId < num_seeds);
  // cache the curand state
  curandState localState = states[threadId];
  for (size_t task_idx = threadId; task_idx < num_task; task_idx += max_span) {
    const IdType rid = input[task_idx / fanout];
    const IdType off = indptr[rid];
    const IdType len = indptr[rid + 1] - indptr[rid];
    size_t k = curand(&localState) % len;
    tmp_src[task_idx] = rid;
    tmp_dst[task_idx] = indices[off + k];
  }
  // restore the state
  states[threadId] = localState;
}

__global__ void init_prefix_num(IdType *src, IdType *dst, 
                                size_t *item_prefix, size_t num_task) {
  size_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
  size_t max_span = blockDim.x * gridDim.x;
  for (size_t task_idx = threadId; task_idx < num_task; task_idx += max_span) {
    if (task_idx == 0) {
      item_prefix[0] = 0;
      continue;
    }
    size_t pre = task_idx - 1;
    if (src[task_idx] == src[pre] && 
            dst[task_idx] == dst[pre]) {
      item_prefix[task_idx] = 0;
    }
    else {
      item_prefix[task_idx] = 1;
    }
  }
}

__global__ void compact_edge(IdType *tmp_src, IdType *tmp_dst,
                             IdType *out_src, IdType *out_dst, 
                             size_t *item_prefix, size_t num_task, 
                             size_t *num_out) {
  size_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
  size_t max_span = blockDim.x * gridDim.x;
  for (size_t task_idx = threadId; task_idx < num_task; task_idx += max_span) {
    out_src[item_prefix[task_idx]] = tmp_src[task_idx];
    out_dst[item_prefix[task_idx]] = tmp_dst[task_idx];
  }
  if (!threadId) {
    *num_out = item_prefix[num_task - 1];
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
 * @param task_key    for profiler data
 * @param states      GPU random seeds list
 * @param num_seeds   GPU the total number of random seeds
 */
void GPUNextdoorSample(const IdType *indptr, const IdType *indices,
                       const IdType *input, const size_t num_input,
                       const size_t fanout, IdType *out_src, IdType *out_dst,
                       size_t *num_out, Context ctx, StreamHandle stream,
                       uint64_t task_key, curandState *states, size_t num_seeds) {
  LOG(DEBUG) << "GPUSample: begin with num_input " << num_input
             << " and fanout " << fanout;
  Timer t_total;
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

  const size_t max_threads = std::min(512ul * 1024, num_seeds);
  size_t num_threads = num_input * fanout;
  if (num_threads > max_threads) {
      num_threads = max_threads;
  }

  const size_t blockSize = Constant::kCudaBlockSize;
  const dim3 grid((num_threads + blockSize - 1) / blockSize);
  const dim3 block(blockSize);
  nextSample<<<grid, block, 0, cu_stream>>>(indptr, indices, input, num_input, fanout,
                                            tmp_src, tmp_dst, states, num_seeds);
  sampler_device->StreamSync(ctx, stream);

  double sample_time = t0.Passed();
  LOG(DEBUG) << "GPUSample: kernel sampling, time cost: "
             << sample_time;


  // sort tmp_src,tmp_dst -> out_src,out_dst, the size is num_input * fanout
  Timer t1;
  size_t temp_storage_bytes = 0;
  CUDA_CALL(cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, 
                                            tmp_src, tmp_src, tmp_dst, tmp_dst, 
                                            num_input * fanout, 0, sizeof(IdType) * 8, 
                                            cu_stream));
  sampler_device->StreamSync(ctx, stream);

  void *d_temp_storage  = sampler_device->AllocWorkspace(ctx, temp_storage_bytes);
  CUDA_CALL(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, 
                                            tmp_src, tmp_src, tmp_dst, tmp_dst, 
                                            num_input * fanout, 0, sizeof(IdType) * 8, 
                                            cu_stream));
  sampler_device->StreamSync(ctx, stream);
  sampler_device->FreeWorkspace(ctx, d_temp_storage);
  double sort_results_time = t1.Passed();
  LOG(DEBUG) << "GPUSample: sort the temporary results, time cost: "
             << sort_results_time;

  // count the prefix num
  Timer t2;
  size_t *item_prefix = static_cast<size_t *>(sampler_device->AllocWorkspace(ctx, sizeof(size_t) * num_input * fanout));
  LOG(DEBUG) << "GPUSample: cuda prefix_num malloc "
             << ToReadableSize(sizeof(int) * num_input * fanout);
  init_prefix_num<<<grid, block, 0, cu_stream>>>(tmp_src, tmp_dst, item_prefix, num_input * fanout);
  sampler_device->StreamSync(ctx, stream);

  temp_storage_bytes = 0;
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes,
                                          item_prefix, item_prefix, num_input * fanout,
                                          cu_stream));
  sampler_device->StreamSync(ctx, stream);

  d_temp_storage = sampler_device->AllocWorkspace(ctx, temp_storage_bytes);
  LOG(DEBUG) << "GPUSample: cuda temp_storage for ExclusiveSum malloc "
             << ToReadableSize(temp_storage_bytes);
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                          item_prefix, item_prefix, num_input * fanout,
                                          cu_stream));
  sampler_device->StreamSync(ctx, stream);
  sampler_device->FreeWorkspace(ctx, d_temp_storage);
  double prefix_sum_time = t2.Passed();
  LOG(DEBUG) << "GPUSample: ExclusiveSum time cost: "
             << prefix_sum_time;

#if 0
  size_t check_len = std::min(30ul, num_input * fanout);
  size_t *check_prefix = new size_t[num_input * fanout];
  IdType *check_src = new IdType[check_len];
  IdType *check_dst = new IdType[check_len];
  sampler_device->CopyDataFromTo(item_prefix, 0, check_prefix, 0, sizeof(size_t) * num_input * fanout,
                                 ctx, CPU(), stream);
  sampler_device->CopyDataFromTo(tmp_src, 0, check_src, 0, sizeof(IdType) * check_len,
                                 ctx, CPU(), stream);
  sampler_device->CopyDataFromTo(tmp_dst, 0, check_dst, 0, sizeof(IdType) * check_len,
                                 ctx, CPU(), stream);
  sampler_device->StreamSync(ctx, stream);
  printf("\e[33mcheck src: \e[0m");
  for (int i = 0 ; i < check_len; ++i) {
    printf("%6d ", check_src[i]);
  }
  printf("\n");
  printf("\e[33mcheck dst: \e[0m");
  for (int i = 0 ; i < check_len; ++i) {
    printf("%6d ", check_dst[i]);
  }
  printf("\n");
  printf("\e[33mcheck prefix sum: \e[0m");
  for (int i = 0 ; i < check_len; ++i) {
    printf("%6ld ", check_prefix[i]);
  }
  printf("\n");
  printf("last prefix = %ld\n", check_prefix[num_input * fanout - 1]);
  delete []check_prefix;
  delete []check_src;
  delete []check_dst;
#endif

  // compact edge
  Timer t3;
  compact_edge<<<grid, block, 0, cu_stream>>>(tmp_src, tmp_dst, out_src, out_dst, item_prefix, 
                                              num_input * fanout, num_out);
  sampler_device->StreamSync(ctx, stream);
  double compact_edge_time = t3.Passed();
  LOG(DEBUG) << "GPUSample: compact_edge time cost: "
             << compact_edge_time;

  sampler_device->FreeWorkspace(ctx, item_prefix);
  sampler_device->FreeWorkspace(ctx, tmp_src);
  sampler_device->FreeWorkspace(ctx, tmp_dst);

  // add time to profiler
  Profiler::Get().LogAdd(task_key, kLogL3SampleCooTime, sample_time);
  Profiler::Get().LogAdd(task_key, kLogL3SampleCountEdgeTime, sort_results_time + prefix_sum_time);
  Profiler::Get().LogAdd(task_key, kLogL3SampleCompactEdgesTime, compact_edge_time);

  double total_time = t_total.Passed();
  LOG(DEBUG) << "GPUSample: succeed total time cost: "
             << total_time;
}

} // namespace cuda
} // namespace common
} // namespace samgraph

