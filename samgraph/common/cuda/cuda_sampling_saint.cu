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
#include "cuda_frequency_hashmap.h"
#include "cuda_function.h"
#include "cuda_utils.h"

namespace samgraph {
namespace common {
namespace cuda {

namespace {

inline __device__ size_t to_pos(size_t node_idx, size_t num_walk, size_t walk_len, size_t step_idx, size_t walk_idx) {
  return node_idx * num_walk * walk_len + step_idx * num_walk + walk_idx;
}

__global__ void sample_random_walk(
    const IdType *indptr, const IdType *indices, const IdType *input,
    const size_t num_input, const size_t random_walk_length,
    const double restart_prob, const size_t num_random_walk,
    IdType *frontier, curandState *random_states, size_t num_random_states) {
  size_t thread_id = blockDim.x * blockDim.y * blockIdx.x +
                     blockDim.y * threadIdx.x + threadIdx.y;
  assert(thread_id < num_random_states);
  curandState local_state = random_states[thread_id];

  size_t node_idx = blockIdx.x * blockDim.y + threadIdx.y;
  const size_t stride = blockDim.y * gridDim.x;
  /** SXN: this loop is also useless*/
  while (node_idx < num_input) {
    IdType start_node = input[node_idx];
    size_t random_walk_idx = threadIdx.x;
    while (random_walk_idx < num_random_walk) {
      IdType node = start_node;
      size_t pos = to_pos(node_idx, num_random_walk, random_walk_length, 0, random_walk_idx);
      frontier[pos] = node;
      for (size_t step_idx = 1; step_idx <= random_walk_length; step_idx++) {
        /*
         *  Get the position on the output position of random walk
         *  suppose that num_random_walk = 2, num_input = 2
         *
         *  layout:
         *     [root step of walk 0 of node 0]
         *     [root step of walk 1 of node 0]
         *     [first step of walk 0 of node 0]
         *     [first step of walk 1 of node 0]
         *     [second step of walk 0 of node 0]
         *     [second step of walk 1 of node 0]
         *     [root step of walk 0 of node 1]
         *     [root step of walk 1 of node 1]
         *     [first step of walk 0 of node 1]
         *     [first step of walk 1 of node 1]
         *     [second step of walk 0 of node 1]
         *     [second step of walk 1 of node 1]
         *     ......
         */
        // size_t pos = node_idx * num_random_walk * random_walk_length +
        //              step_idx * num_random_walk + random_walk_idx;
        size_t pos = to_pos(node_idx, num_random_walk, random_walk_length, step_idx, random_walk_idx);

        if (node != Constant::kEmptyKey) {
          const IdType off = indptr[node];
          const IdType len = indptr[node + 1] - indptr[node];
          if (len == 0) {
            node = Constant::kEmptyKey;
            frontier[pos] = start_node; // avoid compact!
          } else {
            size_t k = curand(&local_state) % len;
            node = indices[off + k];
            frontier[pos] = node;
          }
        } else {
          frontier[pos] = start_node;
        }
        // terminate
        if (restart_prob != 0) {
          if (curand_uniform_double(&local_state) < restart_prob) {
            node = Constant::kEmptyKey;
          }
        }
      }

      random_walk_idx += blockDim.x;
    }

    node_idx += stride;
  }
  // restore the state
  random_states[thread_id] = local_state;
}

}  // namespace

void GPUSampleSaintWalk(const IdType *indptr, const IdType *indices,
                         const IdType *input, const size_t num_input,
                         const size_t random_walk_length,
                         const double random_walk_restart_prob,
                         const size_t num_random_walk,
                         IdType *out_dst,
                        //  IdType *out_data,
                         size_t *num_out,
                         Context ctx, StreamHandle stream,
                         GPURandomStates *random_states, uint64_t task_key) {
  auto sampler_device = Device::Get(ctx);
  auto cu_stream = static_cast<cudaStream_t>(stream);
  size_t num_samples = num_input * num_random_walk * (random_walk_length + 1);

  // 1. random walk sampling
  Timer t0;
  // IdType *tmp_dst = static_cast<IdType *>(sampler_device->AllocWorkspace(ctx, sizeof(IdType) * num_samples));

  dim3 block(Constant::kCudaBlockSize, 1);
  while (static_cast<size_t>(block.x) >= 2 * num_random_walk) {
    block.x /= 2;
    block.y *= 2;
  }
  const dim3 grid(RoundUpDiv(num_input, static_cast<size_t>(block.y)));

  sample_random_walk<<<grid, block, 0, cu_stream>>>(
      indptr, indices, input, num_input, random_walk_length,
      random_walk_restart_prob, num_random_walk, out_dst,
      random_states->GetStates(), random_states->NumStates());

  sampler_device->CopyDataFromTo(&num_samples, 0, num_out, 0, sizeof(size_t),
                                  CPU(), ctx, stream);
  sampler_device->StreamSync(ctx, stream);
  double random_walk_sampling_time = t0.Passed();
  // 2. Only a simple dedup is required?

  // // 2. TopK
  // Timer t1;
  // frequency_hashmap->GetTopK(tmp_src, tmp_dst, num_samples, input, num_input, K,
  //                            out_src, out_dst, out_data, num_out, stream,
  //                            task_key);

  // sampler_device->FreeWorkspace(ctx, tmp_dst);
  // double topk_time = t1.Passed();

  Profiler::Get().LogStepAdd(task_key, kLogL3RandomWalkSampleCooTime,
                             random_walk_sampling_time);
  // Profiler::Get().LogStepAdd(task_key, kLogL3RandomWalkTopKTime, topk_time);
}

}  // namespace cuda
}  // namespace common
}  // namespace samgraph
