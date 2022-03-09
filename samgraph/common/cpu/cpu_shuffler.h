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

#ifndef SAMGRAPH_CPU_SHUFFLER_H
#define SAMGRAPH_CPU_SHUFFLER_H

#include <limits>
#include <memory>

#include "../common.h"

namespace samgraph {
namespace common {

class CPUShuffler : public Shuffler {
 public:
  CPUShuffler(TensorPtr input, int num_epoch, size_t batch_size,
              bool drop_last);
  TensorPtr GetBatch(StreamHandle stream = nullptr);

  uint64_t Epoch() { return _cur_epoch; }
  uint64_t Step() { return _cur_step; }

  size_t NumEpoch() { return _num_epoch; }
  size_t NumStep() { return _num_step; }

 private:
  bool _drop_last;
  bool _initialized;

  uint64_t _cur_epoch;
  uint64_t _cur_step;

  size_t _num_epoch;
  size_t _num_step;

  TensorPtr _data;
  size_t _num_data;

  size_t _batch_size;
  size_t _last_batch_size;

  void ReShuffle();
};

}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_CPU_SHUFFLER_H
