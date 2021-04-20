#include "cpu_permutator.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <random>

#include "../logging.h"

namespace samgraph {
namespace common {

CpuPermutator::CpuPermutator(TensorPtr input, int num_epoch, size_t batch_size,
                             bool drop_last) {
  size_t num_element = input->shape().front();
  CHECK_EQ(input->shape().size(), 1);

  _num_epoch = num_epoch;
  _cur_epoch = -1;

  _input = input;
  _input_size = num_element;
  _drop_last = drop_last;
  _num_step = drop_last ? (num_element / batch_size)
                        : ((num_element - 1) / batch_size) + 1;
  _batch_size = batch_size;
  _last_batch_size = drop_last ? batch_size : num_element % batch_size;

  _initialized = false;
  _cur_step = _num_step;
}

void CpuPermutator::RePermutate() {
  _cur_epoch++;
  if (!_initialized) {
    _initialized = true;
    _cur_epoch = 0;
  } else {
    _cur_epoch++;
  }

  _cur_step = 0;

  if (_cur_epoch >= _num_epoch) {
    return;
  }

  auto seed = std::chrono::system_clock::now().time_since_epoch().count();
  void *data = _input->mutable_data();

  auto g = std::default_random_engine(seed);

  for (size_t i = _input_size - 1; i > 0; i--) {
    std::uniform_int_distribution<size_t> d(0, i);
    size_t candidate = d(g);
    switch (_input->dtype()) {
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

TensorPtr CpuPermutator::GetBatch(cudaStream_t stream) {
  _cur_step++;

  if (_cur_step >= _num_step) {
    RePermutate();
  }

  if (_cur_epoch >= _num_epoch) {
    return nullptr;
  }

  size_t offset = _cur_step * _batch_size;
  size_t size = _cur_step == (_num_step - 1) ? _last_batch_size : _batch_size;

  return Tensor::CreateCopy1D(_input, offset, {size},
                              "random_permutation_batch", stream);
}

}  // namespace common
}  // namespace samgraph
