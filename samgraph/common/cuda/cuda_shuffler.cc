#include "cuda_shuffler.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <random>

#include "../device.h"
#include "../logging.h"
#include "../run_config.h"
#include "cuda_engine.h"
#include "cuda_function.h"

namespace samgraph {
namespace common {
namespace cuda {

GPUShuffler::GPUShuffler(TensorPtr input, size_t num_epoch, size_t batch_size,
                         bool drop_last) {
  _num_data = input->Shape().front();
  CHECK_EQ(input->Shape().size(), 1);
  CHECK_GT(batch_size, 0);

  _num_epoch = num_epoch;
  _cur_epoch = 0;

  _data = input;
  _gpu_data =
      Tensor::Empty(input->Type(), input->Shape(),
                    Engine::Get()->GetSamplerCtx(), "cuda_shuffler_dev_input");

  _drop_last = drop_last;
  _batch_size = batch_size;
  if (drop_last) {
    _num_step = _num_data / batch_size;
    _last_batch_size = batch_size;
  } else {
    _num_step = (_num_data + batch_size - 1) / batch_size;
    _last_batch_size =
        _num_data % batch_size == 0 ? batch_size : _num_data % batch_size;
  }

  _initialized = false;
  _cur_step = _num_step;

  if (RunConfig::option_sanity_check) {
    auto ctx = Engine::Get()->GetSamplerCtx();
    auto device = Device::Get(ctx);
    auto num_node = Engine::Get()->GetGraphDataset()->num_node;
    StreamHandle stream = GPUEngine::Get()->GetSamplerCopyStream();
    auto cu_stream = static_cast<cudaStream_t>(stream);
    _sanity_check_map = static_cast<IdType *>(
        device->AllocDataSpace(ctx, num_node * sizeof(IdType)));
    CUDA_CALL(cudaMemsetAsync(_sanity_check_map, 0, sizeof(IdType) * num_node, cu_stream));
    device->StreamSync(ctx, stream);
  }
}

void GPUShuffler::ReShuffle(StreamHandle stream) {
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

  auto seed = std::chrono::system_clock::now().time_since_epoch().count();
  void *data = _data->MutableData();

  auto g = std::default_random_engine(2022);

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
  device->CopyDataFromTo(_data->Data(), 0, _gpu_data->MutableData(), 0,
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

TensorPtr GPUShuffler::GetBatch(StreamHandle stream) {
  _cur_step++;
  if (_cur_step >= _num_step) {
    ReShuffle(stream);
  }

  if (_cur_epoch >= _num_epoch) {
    return nullptr;
  }

  size_t offset = _cur_step * _batch_size;
  size_t size = _cur_step == (_num_step - 1) ? _last_batch_size : _batch_size;

  auto tensor =
      Tensor::Copy1D(_gpu_data, offset, {size}, "cuda_shuffler_batch", stream);

  if (RunConfig::option_sanity_check) {
    LOG(INFO) << "Doing batch sanity check";
    GPUSanityCheckList(static_cast<const IdType *>(tensor->Data()), size,
                       Constant::kEmptyKey, tensor->Ctx(), stream);
    GPUBatchSanityCheck(_sanity_check_map,
                        static_cast<const IdType *>(tensor->Data()), size,
                        tensor->Ctx(), stream);
  }

  return tensor;
}

}  // namespace cuda
}  // namespace common
}  // namespace samgraph
