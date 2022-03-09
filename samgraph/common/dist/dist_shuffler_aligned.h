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

#ifndef SAMGRAPH_DIST_SHUFFLER_ALIGNED_H
#define SAMGRAPH_DIST_SHUFFLER_ALIGNED_H

#include <limits>
#include <memory>

#include "../common.h"

namespace samgraph {
namespace common {
namespace dist {

class DistAlignedShuffler : public Shuffler {
 public:
  // drop last is disable
  DistAlignedShuffler(TensorPtr input, size_t num_epoch, size_t batch_size,
                      size_t worker_id, size_t num_worker);

  TensorPtr GetBatch(StreamHandle stream = nullptr) override;

  uint64_t Epoch() override { return _cur_epoch; }
  // Global step
  uint64_t Step() override { return _global_step_offset + _cur_local_step; }
  size_t NumEpoch() override { return _num_epoch; }
  size_t NumStep() override { return _num_global_step; }
  size_t NumLocalStep() override { return _num_local_step; }

 private:
  bool _initialized;

  TensorPtr _data;
  TensorPtr _gpu_data;
  size_t _num_data;

  size_t _batch_size;
  size_t _last_batch_size;

  size_t _num_epoch;
  size_t _num_global_step;
  size_t _num_local_step;  // should be same among all the trainer

  uint64_t _cur_epoch;
  uint64_t _cur_local_step;

  // the offset of train set for this sampler
  uint64_t _global_step_offset;
  size_t _global_data_offset;

  void ReShuffle(StreamHandle stream = nullptr);
};

}  // namespace dist
}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_DIST_SHUFFLER_ALIGNED_H