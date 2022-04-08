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

#ifndef SAMGRAPH_RANDOM_STATES_H
#define SAMGRAPH_RANDOM_STATES_H

#include <curand_kernel.h>

#include <vector>

#include "../common.h"
#include "../constant.h"

namespace samgraph {
namespace common {
namespace cuda {

class GPURandomStates {
 public:
  GPURandomStates(SampleType sample_type, const std::vector<size_t>& fanout,
                  const size_t batch_size, Context ctx);
  ~GPURandomStates();

  curandState* GetStates() { return _states; };
  size_t NumStates() { return _num_states; };

 private:
  curandState* _states;
  size_t _num_states;
  Context _ctx;
};

}  // namespace cuda
}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_RANDOM_STATES_H
