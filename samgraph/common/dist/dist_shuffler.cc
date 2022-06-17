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

#include "dist_shuffler.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <random>

#include "../device.h"
#include "../logging.h"
#include "../run_config.h"
#include "dist_engine.h"
#include "../cuda/cuda_function.h"

namespace samgraph {
namespace common {
namespace dist {

DistShuffler::DistShuffler(TensorPtr input, size_t num_epoch, size_t batch_size,
                         int sampler_id, int num_sampler, int num_trainer,
                         bool drop_last) {
  _num_data = input->Shape().front();
  CHECK_EQ(input->Shape().size(), 1);
  CHECK_GT(batch_size, 0);
  CHECK_GT(_num_data, 0);

  _num_epoch = num_epoch;
  _cur_epoch = 0;

  _data = input;

  _drop_last = drop_last;
  _batch_size = batch_size;
  // calculate global num step, but store in _num_step
  if (drop_last) {
    _num_step = _num_data / batch_size;
  } else {
    _num_step = (_num_data + batch_size - 1) / batch_size;
  }
  _epoch_step = _num_step;

  /* Simply _num_step / num_sampler would results in imbalance in smapler
   *    i.e. 15 step, 4 sampler -> 3,3,3,6
   * Solution: spread the unaligned steps to the front sampler:
   *    i.e. 15 step, 4 sampler -> 4,4,4,3
   *                               ^ ^ ^|^
   *                       large_sampler|small_sampler
   * if all equal step, then all is small sampler
   */
  const size_t num_large_sampler = _epoch_step % num_sampler;
  const size_t small_sampler_num_step = _epoch_step / num_sampler;
  const size_t large_sampler_num_step = small_sampler_num_step + 1;
  auto is_large_sampler = [num_large_sampler](size_t sampler_id) 
      {return sampler_id < num_large_sampler; };

  /* tricky thing is how to find how many step belong to previous sampler
   */
  auto get_dataset_offset = [&](size_t sampler_id) {
    if (is_large_sampler(sampler_id)) {
      // if large sampler, all previous samplers are also large sampler
      return batch_size * (large_sampler_num_step * sampler_id);
    } else {
      // if small sampler, all large samplers are at front
      return batch_size * (small_sampler_num_step * sampler_id + num_large_sampler);
    }
  };

  _num_step = is_large_sampler(sampler_id) ? large_sampler_num_step : small_sampler_num_step;
  _dataset_offset = get_dataset_offset(sampler_id);
  _num_local_data = _num_step * batch_size; // let's care about last batch later
  CHECK_EQ(_dataset_offset + _num_local_data, get_dataset_offset(sampler_id + 1));
  CHECK_EQ(_epoch_step * batch_size, get_dataset_offset(num_sampler));

  // now take care that the last batch may be not full
  if (sampler_id == (num_sampler - 1)) {
    _num_local_data = std::min(_num_local_data, _num_data - _dataset_offset);
  }
  _gpu_data =
      Tensor::Empty(input->Type(), {_num_local_data},
                    Engine::Get()->GetSamplerCtx(),
                    "cuda_shuffler_dev_input_" + std::to_string(sampler_id));

  _initialized = false;
  _cur_step = _num_step;

  if (RunConfig::option_sanity_check) {
    auto ctx = Engine::Get()->GetSamplerCtx();
    auto device = Device::Get(ctx);
    StreamHandle stream = DistEngine::Get()->GetSamplerCopyStream();
    auto cu_stream = static_cast<cudaStream_t>(stream);
    auto num_node = Engine::Get()->GetGraphDataset()->num_node;
    _sanity_check_map = static_cast<IdType *>(
        device->AllocDataSpace(ctx, num_node * sizeof(IdType)));
    CUDA_CALL(cudaMemsetAsync(_sanity_check_map, 0, sizeof(IdType) * num_node, cu_stream));
    device->StreamSync(ctx, stream);
  }
}

void DistShuffler::ReShuffle(StreamHandle stream) {
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
    std::uniform_int_distribution<size_t> d(i, _num_data - 1);
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
  device->CopyDataFromTo(_data->Data(), sizeof(IdType) * _dataset_offset, _gpu_data->MutableData(), 0,
                         _gpu_data->NumBytes(), _data->Ctx(), _gpu_data->Ctx(),
                         stream);

  if (RunConfig::option_sanity_check) {
    auto num_node = Engine::Get()->GetGraphDataset()->num_node;
    auto cu_stream = static_cast<cudaStream_t>(stream);
    CUDA_CALL(cudaMemsetAsync(_sanity_check_map, 0, sizeof(IdType) * num_node,
                              cu_stream));
  }

  device->StreamSync(_gpu_data->Ctx(), stream);
}

TensorPtr DistShuffler::GetBatch(StreamHandle stream) {
  _cur_step++;
  if (_cur_step >= _num_step) {
    ReShuffle(stream);
  }

  if (_cur_epoch >= _num_epoch) {
    return nullptr;
  }

  size_t offset = _cur_step * _batch_size;
  CHECK(offset < _num_local_data);
  size_t size = (offset + _batch_size > _num_local_data) ? (_num_local_data - offset) : _batch_size;
  auto tensor =
      Tensor::Copy1D(_gpu_data, offset, {size}, "cuda_shuffler_batch", stream);

  if (RunConfig::option_sanity_check) {
    LOG(INFO) << "Doing batch sanity check";
    cuda::GPUSanityCheckList(static_cast<const IdType *>(tensor->Data()), size,
                       Constant::kEmptyKey, tensor->Ctx(), stream);
    cuda::GPUBatchSanityCheck(_sanity_check_map,
                        static_cast<const IdType *>(tensor->Data()), size,
                        tensor->Ctx(), stream);
  }

  return tensor;
}

}  // namespace dist
}  // namespace common
}  // namespace samgraph
