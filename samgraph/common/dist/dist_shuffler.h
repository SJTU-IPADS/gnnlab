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

#ifndef SAMGRAPH_DIST_SHUFFLER_H
#define SAMGRAPH_DIST_SHUFFLER_H

#include <limits>
#include <memory>

#include "../common.h"

namespace samgraph {
namespace common {
namespace dist {

class DistShuffler : public Shuffler {
 public:
  DistShuffler(TensorPtr input, size_t num_epoch, size_t batch_size,
              int sampler_id, int num_sampler, int num_trainer,
              bool drop_last);
  TensorPtr GetBatch(StreamHandle stream = nullptr) override;

  uint64_t Epoch() override { return _cur_epoch; }
  uint64_t Step() override {
    return (_dataset_offset / _batch_size) + _cur_step;
  }

  size_t NumEpoch() override { return _num_epoch; }
  // return the total steps for each epoch
  // reasons: profiler needs this to create total space
  size_t NumStep() override { return _epoch_step; }
  size_t NumLocalStep() override { return _num_step; }
  bool IsLastBatch() { return _cur_step == (_num_step - 1); }

  void Reset() {
    _cur_step = _num_step;
    _cur_epoch = 0;
    _initialized = false;
  }

 private:
  bool _drop_last;
  bool _initialized;

  uint64_t _cur_epoch;
  uint64_t _cur_step;

  size_t _num_epoch;
  // number of steps for this sampler
  size_t _num_step;
  // total steps each epoch
  size_t _epoch_step;
  // the offset of train set for this sampler
  size_t _dataset_offset;

  // local copy of full train set. currently is at cpu
  TensorPtr _data;
  TensorPtr _gpu_data;
  // size of actually used global train set
  size_t _num_data;
  // aligned up to num worker, but not batch size
  size_t _num_local_data;

  size_t _batch_size;

  IdType *_sanity_check_map;

  void ReShuffle(StreamHandle stream = nullptr);
};

}  // namespace dist
}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_DIST_SHUFFLER_H
