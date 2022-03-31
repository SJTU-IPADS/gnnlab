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

#include "dist_shuffler_aligned.h"

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

DistAlignedShuffler::DistAlignedShuffler(TensorPtr input, size_t num_epoch,
                                         size_t batch_size, size_t worker_id,
                                         size_t num_worker) {
  CHECK_EQ(input->Shape().size(), 1);
  CHECK_EQ(input->Ctx(), CPU());
  CHECK_GT(batch_size, 0);

  _initialized = false;

  size_t origin_num_data = input->Shape().front();
  // aligned to num_worker
  _num_data = RoundUp(origin_num_data, num_worker);
  _data = Tensor::Empty(input->Type(), {_num_data}, CPU(),
                        "DistAlignedShuffler data");
  Device::Get(CPU())->CopyDataFromTo(input->Data(), 0, _data->MutableData(), 0,
                                     input->NumBytes(), CPU(), CPU());
  for (size_t i = 0; i < (_num_data - origin_num_data); i++) {
    const IdType *src = static_cast<const IdType *>(input->MutableData()) + i;
    IdType *dst =
        static_cast<IdType *>(_data->MutableData()) + i + origin_num_data;
    *dst = *src;
  }

  _num_epoch = num_epoch;
  size_t num_data_per_worker = _num_data / num_worker;
  _num_local_step = RoundUpDiv(num_data_per_worker, batch_size);
  _num_global_step = _num_local_step * num_worker;

  _global_step_offset = _num_local_step * worker_id;
  _global_data_offset = num_data_per_worker * worker_id;

  _cur_epoch = 0;
  _cur_local_step = _num_local_step;

  _batch_size = batch_size;
  _last_batch_size = num_data_per_worker % _batch_size == 0
                         ? _batch_size
                         : num_data_per_worker % _batch_size;

  _gpu_data = Tensor::Empty(
      input->Type(), {num_data_per_worker}, Engine::Get()->GetSamplerCtx(),
      "cuda_shuffler_dev_input_" + std::to_string(worker_id));
}

void DistAlignedShuffler::ReShuffle(StreamHandle stream) {
  if (!_initialized) {
    _cur_epoch = 0;
    _initialized = true;
  } else {
    _cur_epoch++;
  }

  _cur_local_step = 0;

  if (_cur_epoch >= _num_epoch) {
    return;
  }

  // auto seed = std::chrono::system_clock::now().time_since_epoch().count();
  // let all samplers have the same seed
  auto seed = _cur_epoch;
  void *data = _data->MutableData();

  auto g = std::default_random_engine(seed);

  for (size_t i = _num_data - 1; i > 0; i--) {
    std::uniform_int_distribution<size_t> d(0, i);
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
  device->CopyDataFromTo(_data->Data(), sizeof(IdType) * _global_data_offset,
                         _gpu_data->MutableData(), 0, _gpu_data->NumBytes(),
                         _data->Ctx(), _gpu_data->Ctx(), stream);

  device->StreamSync(_gpu_data->Ctx(), stream);
}

TensorPtr DistAlignedShuffler::GetBatch(StreamHandle stream) {
  _cur_local_step++;
  if (_cur_local_step >= _num_local_step) {
    ReShuffle(stream);
  }

  if (_cur_epoch >= _num_epoch) {
    return nullptr;
  }

  size_t offset = _cur_local_step * _batch_size;
  size_t size =
      _cur_local_step == (_num_local_step - 1) ? _last_batch_size : _batch_size;

  auto tensor =
      Tensor::Copy1D(_gpu_data, offset, {size}, "cuda_shuffler_batch", stream);

  return tensor;
}

}  // namespace dist
}  // namespace common
}  // namespace samgraph
