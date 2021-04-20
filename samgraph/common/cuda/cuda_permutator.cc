#include "cuda_permutator.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <random>

#include "../logging.h"
#include "cuda_engine.h"

namespace samgraph {
namespace common {
namespace cuda {

CudaPermutator::CudaPermutator(TensorPtr input, size_t num_epoch,
                               size_t batch_size, bool drop_last) {
  _input_size = input->shape().front();
  CHECK_EQ(input->shape().size(), 1);

  _num_epoch = num_epoch;
  _cur_epoch = 0;

  _input = input;
  _dev_input = Tensor::Empty(input->dtype(), input->shape(),
                             Engine::Get()->GetSampleDevice(),
                             "cuda_permutator d_input");

  _drop_last = drop_last;
  _num_step = drop_last ? (_input_size / batch_size)
                        : ((_input_size - 1) / batch_size) + 1;
  _batch_size = batch_size;
  _last_batch_size = drop_last ? batch_size : _input_size % batch_size;

  _initialized = false;
  _cur_step = _num_step;
}

void CudaPermutator::RePermutate(cudaStream_t stream) {
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

  if (stream) {
    CUDA_CALL(cudaMemcpyAsync(_dev_input->mutable_data(), _input->data(),
                              _dev_input->size(), cudaMemcpyHostToDevice,
                              stream));
    CUDA_CALL(cudaStreamSynchronize(stream));
  } else {
    CUDA_CALL(cudaMemcpy(_dev_input->mutable_data(), _input->data(),
                         _dev_input->size(), cudaMemcpyHostToDevice));
  }
}

TensorPtr CudaPermutator::GetBatch(cudaStream_t stream) {
  _cur_step++;
  if (_cur_step >= _num_step) {
    RePermutate(stream);
  }

  if (_cur_epoch >= _num_epoch) {
    return nullptr;
  }

  size_t offset = _cur_step * _batch_size;
  size_t size = _cur_step == (_num_step - 1) ? _last_batch_size : _batch_size;

  return Tensor::CreateCopy1D(_dev_input, offset, {size},
                              "random_permutation_batch", stream);
}

}  // namespace cuda
}  // namespace common
}  // namespace samgraph
