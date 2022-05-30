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

#include <cassert>
#include <chrono>
#include <numeric>

#include "../common.h"
#include "../constant.h"
#include "../device.h"
#include "../logging.h"
#include "../run_config.h"
#include "../timer.h"
#include "cuda_random_states.h"

namespace samgraph {
namespace common {
namespace cuda {

namespace {

__global__ void init_random_states(curandState *states, size_t num,
                                   unsigned long seed) {
  size_t threadId = threadIdx.x + blockIdx.x * blockDim.x;
  if (threadId < num) {
    /** Using different seed & constant sequence 0 can reduce memory 
      * consumption by 800M
      * https://docs.nvidia.com/cuda/curand/device-api-overview.html#performance-notes
      */
    curand_init(seed+threadId, 0, 0, &states[threadId]);
  }
}

size_t PredictRandomWalkMaxThreads(size_t num_nodes, size_t num_random_walk) {
  size_t block_x = Constant::kCudaBlockSize;
  size_t block_y = 1;

  while (block_x >= 2 * num_random_walk) {
    block_x /= 2;
    block_y *= 2;
  }

  size_t grid_x = RoundUpDiv(num_nodes, block_y);

  return grid_x * block_x * block_y;
}

}  // namespace

GPURandomStates::GPURandomStates(SampleType sample_type,
                                 const std::vector<size_t> &fanout,
                                 const size_t batch_size, Context ctx) {
  _ctx = ctx;
  auto device = Device::Get(_ctx);

  switch (sample_type) {
    case kKHop0:
      _num_states = PredictNumNodes(batch_size, fanout, fanout.size() - 1);
      break;
    case kKHop1:
      _num_states = PredictNumNodes(batch_size, fanout, fanout.size());
      _num_states = Min(_num_states, Constant::kKHop1MaxThreads);
      break;
    case kWeightedKHop:
    case kWeightedKHopPrefix:
      _num_states = PredictNumNodes(batch_size, fanout, fanout.size());
      _num_states = Min(_num_states, Constant::kWeightedKHopMaxThreads);
      break;
    case kSaint:
    case kRandomWalk:
      _num_states = PredictRandomWalkMaxThreads(
          PredictNumNodes(batch_size, fanout, fanout.size() - 1),
          RunConfig::num_random_walk);
      break;
    case kKHop2:
      _num_states = PredictNumNodes(batch_size, fanout, fanout.size() - 1);
      break;
    case kWeightedKHopHashDedup:
      _num_states = PredictNumNodes(batch_size, fanout, fanout.size() - 1);
      break;
    default:
      CHECK(0);
  }

  _states = static_cast<curandState *>(
      device->AllocDataSpace(_ctx, sizeof(curandState) * _num_states));

  const dim3 grid(
      RoundUpDiv(_num_states, static_cast<size_t>(Constant::kCudaBlockSize)));
  const dim3 block(Constant::kCudaBlockSize);

  unsigned long seed =
      std::chrono::system_clock::now().time_since_epoch().count();
  init_random_states<<<grid, block>>>(_states, _num_states, seed);
}

GPURandomStates::~GPURandomStates() {
  auto device = Device::Get(_ctx);
  device->FreeDataSpace(_ctx, _states);
}

}  // namespace cuda
}  // namespace common
}  // namespace samgraph
