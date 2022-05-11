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

#pragma once

#include "../common.h"
#include "../constant.h"
#include "../cuda/cuda_shuffler.h"
namespace samgraph {
namespace common {
namespace dist {

class PreSampler {
 public:
  PreSampler(TensorPtr input, size_t batch_size, size_t num_nodes);
  ~PreSampler();
  void DoPreSample();
  TensorPtr GetFreq();
  void GetFreq(IdType*);
  TensorPtr GetRankNode();
  void GetRankNode(TensorPtr &);
  void GetRankNode(IdType *);
  static inline void SetSingleton(PreSampler* p) { singleton = p; }
  static inline PreSampler* Get() { return singleton; }
  TaskPtr DoPreSampleShuffle();
 private:
  Id64Type * freq_table;
  // TensorPtr freq_table;
  size_t _num_nodes, _num_step;
  static PreSampler* singleton;
  cuda::GPUShuffler* _shuffler;

};

}
}
}
