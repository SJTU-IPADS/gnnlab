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

__global__ void sample_random_walk(
    const IdType *indptr, const IdType *indices, const IdType *input,
    const size_t num_input, const size_t random_walk_length,
    const double restart_prob, const size_t num_random_walk, IdType *tmp_src,
    IdType *tmp_dst, curandState *random_states, size_t num_random_states) {
  size_t thread_id = blockDim.x * blockDim.y * blockIdx.x +
                     blockDim.y * threadIdx.x + threadIdx.y;
  assert(thread_id < num_random_states);
  curandState local_state = random_states[thread_id];

  size_t node_idx = blockIdx.x * blockDim.y + threadIdx.y;
  const size_t stride = blockDim.y * gridDim.x;

  while (node_idx < num_input) {
    IdType start_node = input[node_idx];
    size_t random_walk_idx = threadIdx.x;
    while (random_walk_idx < num_random_walk) {
      IdType node = start_node;
      for (size_t step_idx = 0; step_idx < random_walk_length; step_idx++) {
        /*
         *  Get the position on the output position of random walk
         *  suppose that num_random_walk = 3, num_input = 2
         *
         *  layout:
         *     [first step of walk 0 of node 0]
         *     [first step of walk 0 of node 0]
         *     [first step of walk 0 of node 0]
         *     [first step of walk 0 of node 1]
         *     [first step of walk 0 of node 1]
         *     [first step of walk 0 of node 1]
         *     [second step of walk 0 of node 0]
         *     [second step of walk 0 of node 0]
         *     [second step of walk 0 of node 0]
         *     [second step of walk 0 of node 0]
         *     [second step of walk 0 of node 0]
         *     [second step of walk 0 of node 0]
         */
        size_t pos = step_idx * num_input * num_random_walk +
                     node_idx * num_random_walk + random_walk_idx;
        if (node == Constant::kEmptyKey) {
          tmp_src[pos] = Constant::kEmptyKey;
        } else {
          const IdType off = indptr[node];
          const IdType len = indptr[node + 1] - indptr[node];

          if (len == 0) {
            tmp_src[pos] = Constant::kEmptyKey;
            node = Constant::kEmptyKey;
          } else {
            size_t k = curand(&local_state) % len;
            tmp_src[pos] = start_node;
            tmp_dst[pos] = indices[off + k];
            node = indices[off + k];

            // terminate
            if (curand_uniform_double(&local_state) < restart_prob) {
              node = Constant::kEmptyKey;
            }
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

void GPUSampleRandomWalk(const IdType *indptr, const IdType *indices,
                         const IdType *input, const size_t num_input,
                         const size_t random_walk_length,
                         const double random_walk_restart_prob,
                         const size_t num_random_walk, const size_t K,
                         IdType *out_src, IdType *out_dst, IdType *out_data,
                         size_t *num_out, FrequencyHashmap *frequency_hashmap,
                         Context ctx, StreamHandle stream,
                         GPURandomStates *random_states, uint64_t task_key) {
  auto sampler_device = Device::Get(ctx);
  auto cu_stream = static_cast<cudaStream_t>(stream);
  size_t num_samples = num_input * num_random_walk * random_walk_length;

  // 1. random walk sampling
  IdType *tmp_src = static_cast<IdType *>(
      sampler_device->AllocWorkspace(ctx, sizeof(IdType) * num_samples));
  IdType *tmp_dst = static_cast<IdType *>(
      sampler_device->AllocWorkspace(ctx, sizeof(IdType) * num_samples));

  dim3 block(Constant::kCudaBlockSize, 1);
  while (static_cast<size_t>(block.x) >= 2 * num_random_walk) {
    block.x /= 2;
    block.y *= 2;
  }
  const dim3 grid(RoundUpDiv(num_input, static_cast<size_t>(block.y)));

  sample_random_walk<<<grid, block, 0, cu_stream>>>(
      indptr, indices, input, num_input, random_walk_length,
      random_walk_restart_prob, num_random_walk, tmp_src, tmp_dst,
      random_states->GetStates(), random_states->NumStates());
  sampler_device->StreamSync(ctx, stream);

  frequency_hashmap->Reset(stream);
  frequency_hashmap->GetTopK(tmp_src, tmp_dst, num_samples, input, num_input, K,
                             out_src, out_dst, out_data, num_out, stream);
}

}  // namespace cuda
}  // namespace common
}  // namespace samgraph
