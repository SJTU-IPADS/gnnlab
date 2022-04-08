#include "model.h"
#include <cassert>

namespace samgraph {
namespace sam_backend {

namespace {

template <typename T, size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void add_kernel(const T *input0, const T *input1, T *output, size_t size) {

  assert(BLOCK_SIZE == blockDim.x);

  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);
#pragma unroll
  for (size_t i = threadIdx.x + block_start; i < block_end; i += BLOCK_SIZE) {
    if (i < size) {
      output[i] = input0[i] + input1[i];
    }
  }
}
template <typename T, size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void mul_kernel(const T *input0, const T *input1, T *output, size_t size) {

  assert(BLOCK_SIZE == blockDim.x);

  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);
#pragma unroll
  for (size_t i = threadIdx.x + block_start; i < block_end; i += BLOCK_SIZE) {
    if (i < size) {
      output[i] = input0[i] * input1[i];
    }
  }
}

template <typename T>
__host__ void forward_task(Element *op, const T *input0, const T *input1, T *output,
                           const size_t num_input) {
  const size_t num_tiles = RoundUpDiv(num_input, Constant::kCudaTileSize);
  const dim3 grid(num_tiles);
  const dim3 block(Constant::kCudaBlockSize);
  switch (op->elementType) {
  case kEwTypeAdd:
    add_kernel<T, Constant::kCudaBlockSize, Constant::kCudaTileSize>
        <<<grid, block, 0, op->_model->stream>>>(input0, input1, output, num_input);
    break;
  case kEwTypeMul:
    mul_kernel<T, Constant::kCudaBlockSize, Constant::kCudaTileSize>
        <<<grid, block, 0, op->_model->stream>>>(input0, input1, output, num_input);
    break;
  }
  if (common::RunConfig::option_samback_cuda_launch_blocking) {
    CUDA_CALL(cudaStreamSynchronize(op->_model->stream));
  }
}

template <typename T>
__host__ void backward_task(Element *op, T *input0_grad, T *input1_grad, const T *output_grad,
                            const size_t num_input) {
  // const size_t num_tiles = RoundUpDiv(num_input, Constant::kCudaTileSize);
  // const dim3 grid(num_tiles);
  // const dim3 block(Constant::kCudaBlockSize);
  switch (op->elementType) {
  case kEwTypeAdd: {
    if (input0_grad) {
      cudaMemcpyAsync(input0_grad, output_grad, num_input * sizeof(T), cudaMemcpyDeviceToDevice,
                      op->_model->stream);
    }
    if (input1_grad) {
      cudaMemcpyAsync(input1_grad, output_grad, num_input * sizeof(T), cudaMemcpyDeviceToDevice,
                      op->_model->stream);
    }
    break;
  }
  case kEwTypeMul:
  default:
    assert(false);
  }
  if (common::RunConfig::option_samback_cuda_launch_blocking) {
    CUDA_CALL(cudaStreamSynchronize(op->_model->stream));
  }
}

} // namespace

Element::Element(Model *model, const GradTensorPtr input0, const GradTensorPtr input1,
                 ElementType _elementType)
    : GnnOp(model, input0, input1) {
  this->elementType = _elementType;
  output = GradTensor::Null(input0->RequireGrad() || input1->RequireGrad());
}

void Element::forward() {
  size_t num_input = std::accumulate(inputs[0]->Shape().begin(), inputs[0]->Shape().end(), 1ul,
                                     std::multiplies<size_t>());
  const TrainType *input0_ptr = tensor_cast_const<TrainType>(inputs[0]->data());
  const TrainType *input1_ptr = tensor_cast_const<TrainType>(inputs[1]->data());
  TrainType *output_ptr = tensor_cast<TrainType>(output->data());
  forward_task(this, input0_ptr, input1_ptr, output_ptr, num_input);
}

void Element::backward() {
  if (output->RequireGrad() == false)
    return;
  size_t num_input = inputs[0]->data()->NumItem();
  TrainType *input0_grad = nullptr, *input1_grad = nullptr;
  if (inputs[0]->RequireGrad())
    input0_grad = tensor_cast<TrainType>(inputs[0]->grad());
  if (inputs[1]->RequireGrad())
    input1_grad = tensor_cast<TrainType>(inputs[1]->grad());
  const TrainType *output_grad = tensor_cast_const<TrainType>(output->grad());
  backward_task(this, input0_grad, input1_grad, output_grad, num_input);
}

void Element::prepare() {
  TaskPtr cur_task = _model->cur_task;
  output->Resize(inputs[0]->Type(), inputs[0]->Shape(), _model->ctx, "");
}

bool Element::generate_grad()   const {return true;}
bool Element::accumulate_grad() const {return false;}
std::string Element::name() const {return "element";}

} // namespace sam_backend
} // namespace samgraph