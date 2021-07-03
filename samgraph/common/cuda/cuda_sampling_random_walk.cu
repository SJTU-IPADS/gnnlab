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
inline __device__ size_t get_rw_output_pos(size_t node_idx, size_t rw_idx,
                                           size_t step_idx, size_t num_nodes,
                                           size_t num_rw) {
  return step_idx * num_nodes * num_rw + node_idx * num_rw + rw_idx;
}

__global__ void sample_random_walk(
    const IdType *indptr, const IdType *indices, const IdType *input,
    const size_t num_input, const size_t random_walk_length,
    const double restart_prob, const size_t num_random_walk, IdType *tmp_src,
    IdType *tmp_dst, const size_t tile_size, const size_t block_size,
    curandState *random_states, size_t num_random_states) {
  assert(block_size == blockDim.x);
  assert(num_random_walk == blockDim.y);

  const size_t block_start = tile_size * blockIdx.x;
  const size_t block_end = tile_size * (blockIdx.x + 1);

  size_t thread_id = blockDim.x * blockDim.y * blockIdx.x +
                     blockDim.y * threadIdx.x + threadIdx.y;
  curandState local_state = random_states[thread_id];

  for (size_t node_idx = threadIdx.x + block_start; node_idx < block_end;
       node_idx += block_size) {
    size_t random_walk_idx = threadIdx.y;
    if (node_idx < num_input && random_walk_idx < num_random_walk) {
      const IdType node = input[node_index];
      for (size_t step_idx = 0; step_idx < random_walk_length; step_idx++) {
        size_t pos = get_rw_output_pos(node_idx, random_walk_idx, step_idx,
                                       num_input, num_random_walk);
        if (node == Constant::kEmptyKey) {
          tmp_src[pos] = Constant::kEmptyKey;
          tmp_dst[pos] = Constant::kEmptyKey;
        } else {
          const IdType off = indptr[node];
          const IdType len = indptr[node + 1] - indptr[node];

          if (len == 0) {
            tmp_src[pos] = Constant::kEmptyKey;
            tmp_dst[pos] = Constant::kEmptyKey;
            node = Constant::kEmptyKey;
          } else {
            size_t k = curand(&local_state) % len;
            tmp_src[task_idx] = rid;
            tmp_dst[task_idx] = indices[off + k];
            node = indices[off + k];

            // terminate
            if (curand_uniform_double(&local_state) < restart_prob) {
              node = Constant::kEmptyKey;
            }
          }
        }
      }
    }
  }
  // restore the state
  random_states[thread_id] = local_state;
}

__global__ void count_frequency(IdType *tmp_src, IdType *tmp_dst) {}

__global__ void topk() {}

}  // namespace

void GPUSampleRandomWalk(const IdType *indptr, const IdType *indices,
                         const IdType *input, const size_t num_input,
                         const size_t random_walk_length,
                         const double random_walk_restart_prob,
                         const size_t num_random_walk,
                         const const size_t num_neighbor, IdType *out_src,
                         IdType *out_dst, size_t *num_out, Context ctx,
                         StreamHandle stream, GPURandomStates *random_states,
                         uint64_t task_key) {
  auto sampler_device = Device::Get(ctx);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  // 1. random walk sampling
  IdType *tmp_src = static_cast<IdType *>(sampler_device->AllocWorkspace(
      ctx, sizeof(IdType) * num_input * num_random_walk * random_walk_length));
  IdType *tmp_dst = static_cast<IdType *>(sampler_device->AllocWorkspace(
      ctx, sizeof(IdType) * num_input * num_random_walk * random_walk_length));

  size_t tile_size = Constant::kCudaTileSize / num_random_walk;
  size_t block_size = Constant::kCudaBlockSize / num_random_walk;
  tile_size = tile_size == 0 ? 1 : tile_size;
  block_size = block_size == 0 ? 1 : block_size;

  const size_t num_tiles = RoundUpDiv(num_input, tile_size);
  const dim3 grid(num_tiles);
  const dim3 block(block_size, num_random_walk);

  template sample_random_walk<<<grid, block, 0, cu_stream>>>(
      indptr, indices, input, num_input, random_walk_length,
      random_walk_restart_prob, num_random_walk, tmp_src, tmp_dst, tile_size,
      block_size, random_states->GetStates(), random_states->NumStates());
  sampler_device->StreamSync(ctx, stream);

  double sample_time = t0.Passed();
  LOG(DEBUG) << "GPUSample: kernel sampling, time cost: " << sample_time;
}

}  // namespace cuda
}  // namespace common
}  // namespace samgraph
