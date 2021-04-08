#include <algorithm>
#include <random>
#include <chrono>
#include <iterator>
#include <cstdint>
#include <numeric>

#include "../logging.h"
#include "cuda_engine.h"
#include "cuda_permutator.h"

namespace samgraph {
namespace common {
namespace cuda {

CudaPermutator::CudaPermutator(std::shared_ptr<IdTensor> input, int num_epoch, size_t batch_size, bool drop_last) {
    size_t num_element = input->shape().front();
    SAM_CHECK_EQ(input->shape().size(), 1);

    _num_epoch = num_epoch;
    _cur_epoch = -1;

    _input = input;
    _d_input = Tensor::Empty(input->dtype(), input->shape(), SamGraphEngine::GetEngine()->GetSampleDevice(), "cuda_permutator d_input");
    _num_element = num_element;
    _drop_last = drop_last;
    _num_step = drop_last ? (num_element / batch_size) : ((num_element - 1) / batch_size) + 1;
    _batch_size = batch_size;
    _last_batch_size = drop_last ? batch_size : num_element % batch_size;
    
    _cur_step = std::numeric_limits<int>::max();
}

void CudaPermutator::RePermutate(cudaStream_t stream) {
    _cur_epoch++;
    _cur_step = 0;

    if (_cur_epoch >= _num_epoch) {
        return;
    }

    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    void *data = _input->mutable_data();

    auto g = std::default_random_engine(seed);

    for (size_t i = _num_element - 1; i > 0; i--) {
        std::uniform_int_distribution<size_t> d(0, i);
        size_t candidate = d(g);
        switch(_input->dtype()) {
            case kSamI32:
                std::swap((reinterpret_cast<int *>(data))[i], (reinterpret_cast<int *>(data))[candidate]);
                break;
            case kSamF32:
            case kSamI8:
            case kSamU8:
            case kSamF16:
            case kSamI64:
            case kSamF64:
            default:
                SAM_CHECK(0);
        }
    }

    if (stream) {
        CUDA_CALL(cudaMemcpyAsync(_d_input->mutable_data(), _input->data(), _d_input->size(), cudaMemcpyHostToDevice, stream));
        CUDA_CALL(cudaStreamSynchronize(stream));
    } else {
        CUDA_CALL(cudaMemcpy(_d_input->mutable_data(), _input->data(), _d_input->size(), cudaMemcpyHostToDevice));
    }
}

std::shared_ptr<IdTensor> CudaPermutator::GetBatch(cudaStream_t stream) {
    if (_cur_epoch >= _num_epoch) {
        return nullptr;
    }

    _cur_step++;

    if (_cur_step >= _num_step) {
        RePermutate(stream);
        // _cur_epoch++;
        // _cur_step = 0;
    }

    size_t offset = _cur_step * _batch_size;
    size_t size = _cur_step == (_num_step - 1) ? _last_batch_size : _batch_size;

    return IdTensor::CreateCopy1D(_d_input, offset, {size}, "random_permutation_batch", stream);
}

} // namespace cuda
} // namespace common
} // namespace samgraph
