#include "cpu_shuffler.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <random>

#include "../logging.h"

namespace samgraph {
namespace common {

CPUShuffler::CPUShuffler(TensorPtr input, int num_epoch, size_t batch_size,
                         bool drop_last) {
  _num_data = input->Shape().front();
  CHECK_EQ(input->Shape().size(), 1);
  CHECK_GT(batch_size, 0);

  _num_epoch = num_epoch;
  _cur_epoch = 0;

  _data = input;
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
}

void CPUShuffler::ReShuffle() {
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
}

TensorPtr CPUShuffler::GetBatch(StreamHandle stream) {
  _cur_step++;

  if (_cur_step >= _num_step) {
    ReShuffle();
  }

  if (_cur_epoch >= _num_epoch) {
    return nullptr;
  }

  size_t offset = _cur_step * _batch_size;
  size_t size = _cur_step == (_num_step - 1) ? _last_batch_size : _batch_size;

  LOG(DEBUG) << "Copy shuffled dataset with offset=" << offset << ", size=" << size;

  return Tensor::Copy1D(_data, offset, {size}, "cpu_shuffler_batch");
}

}  // namespace common
}  // namespace samgraph
