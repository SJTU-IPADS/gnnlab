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

#include "dist_shuffler_aligned_trainer.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <random>

#include "../cuda/cuda_function.h"
#include "../device.h"
#include "../logging.h"
#include "../run_config.h"
#include "dist_engine.h"

namespace samgraph {
namespace common {
namespace dist {

DistTrainerAlignedShuffler::DistTrainerAlignedShuffler(TensorPtr input, size_t num_epoch,
                                         size_t batch_size, size_t worker_id,
                                         size_t num_sampler, size_t num_trainer) {
  CHECK_EQ(input->Shape().size(), 1);
  CHECK_EQ(input->Ctx(), CPU());
  CHECK_GT(batch_size, 0);
  _num_epoch = num_epoch;
  _cur_epoch = 0;

  _batch_size = batch_size;

  size_t origin_num_data = input->Shape().front();
  // aligned to num_sampler
  _num_data = RoundUp(origin_num_data, batch_size * LCM(num_sampler, num_trainer));
  CHECK_EQ(_num_data % batch_size, 0);
  _epoch_step = _num_data / batch_size;
  CHECK_EQ(_epoch_step % num_sampler, 0);
  CHECK_EQ(_epoch_step % num_trainer, 0);
  _data = Tensor::Empty(input->Type(), {_num_data}, CPU(),
                        "DistTrainerAlignedShuffler data");
  Device::Get(CPU())->CopyDataFromTo(input->Data(), 0, _data->MutableData(), 0,
                                     input->NumBytes(), CPU(), CPU());
  Device::Get(CPU())->CopyDataFromTo(input->Data(), 0, _data->MutableData(), input->NumBytes(),
                                     _data->NumBytes() - input->NumBytes(), CPU(), CPU());

  _epoch_step = std::min(_epoch_step, RunConfig::step_max_boundary);
  _epoch_step = RoundUp(_epoch_step, LCM(num_sampler, num_trainer));
  CHECK_LE(_epoch_step * batch_size, _num_data);

  _num_step = _epoch_step / num_sampler;
  _num_local_data = _num_step * batch_size;
  // _num_data may be smaller than _data.shape. 
  // only front of _data is shuffled, but candidate is chosen from entire _data
  _num_data = _epoch_step * batch_size;

  _global_step_offset = _num_step * worker_id;
  _dataset_offset = _num_local_data * worker_id;

  _gpu_data = Tensor::Empty(
      input->Type(), {_num_local_data}, Engine::Get()->GetSamplerCtx(),
      "cuda_shuffler_dev_input_" + std::to_string(worker_id));

  _initialized = false;
  _cur_step = _num_step;
}

void DistTrainerAlignedShuffler::ReShuffle(StreamHandle stream) {
  if (!_initialized) {
    _cur_epoch = 0;
    _initialized = true;
  } else {
    _cur_epoch++;
  }

  _cur_step = 0;

  if (_cur_epoch >= _num_epoch) {
    return;
  }

  // auto seed = std::chrono::system_clock::now().time_since_epoch().count();
  // let all samplers have the same seed
  auto seed = _cur_epoch;
  void *data = _data->MutableData();

  auto g = std::default_random_engine(seed);

  for (size_t i = 0; i < _num_data - 1; i++) {
    std::uniform_int_distribution<size_t> d(i, _data->Shape().front() - 1);
    size_t candidate = d(g);
    switch (_data->Type()) {
      case kI32:
        std::swap((reinterpret_cast<int *>(data))[i],
                  (reinterpret_cast<int *>(data))[candidate]);
        break;
      case kF32:
      case kI8:
      case kU8:
      case kF16:
      case kI64:
      case kF64:
      default:
        CHECK(0);
    }
  }

  auto device = Device::Get(_gpu_data->Ctx());
  device->CopyDataFromTo(_data->Data(), sizeof(IdType) * _dataset_offset,
                         _gpu_data->MutableData(), 0, _gpu_data->NumBytes(),
                         _data->Ctx(), _gpu_data->Ctx(), stream);

  device->StreamSync(_gpu_data->Ctx(), stream);
}

TensorPtr DistTrainerAlignedShuffler::GetBatch(StreamHandle stream) {
  _cur_step++;
  if (_cur_step >= _num_step) {
    ReShuffle(stream);
  }

  if (_cur_epoch >= _num_epoch) {
    return nullptr;
  }

  size_t offset = _cur_step * _batch_size;
  CHECK(offset < _num_local_data);
  CHECK(offset + _batch_size <= _num_local_data);
  size_t size = _batch_size;
  // _cur_step is a local step
  // all sampler's first iteration's step, make it larger
  if (_cur_step * RunConfig::num_sample_worker + RunConfig::worker_id < RunConfig::num_train_worker && _cur_epoch == 0) {
    size *= 1.2;
    size = std::min((_num_local_data - offset), size);
  }
  auto tensor =
      Tensor::Copy1D(_gpu_data, offset, {size}, "cuda_shuffler_batch", stream);

  return tensor;
}

}  // namespace dist
}  // namespace common
}  // namespace samgraph
