#include "model.h"
#include <cassert>

namespace samgraph {
namespace sam_backend {

namespace {

template <typename T>
__global__ void add_bias_kernel(const T *input, const T *bias, T *output, size_t size,
                                size_t hidden_dim) {

  size_t v = blockIdx.x * blockDim.y + threadIdx.y;
  const size_t stride = blockDim.y * gridDim.x;

  while (v < size) {
    size_t col = threadIdx.x;
    while (col < hidden_dim) {
      size_t local_idx = v * hidden_dim + col;
      output[local_idx] = input[local_idx] + bias[col];
      col += blockDim.x;
    }
    v += stride;
  }
}

template <typename T>
__host__ void forward_task(Bias *op, const T *input, const T *bias, T *output,
                           const size_t num_input, const size_t hidden_dim) {
  dim3 block(256, 1);
  while (static_cast<size_t>(block.x) >= 2 * hidden_dim) {
    block.x /= 2;
    block.y *= 2;
  }
  const dim3 grid(RoundUpDiv(num_input, static_cast<size_t>(block.y)));
  add_bias_kernel<<<grid, block, 0, op->_model->stream>>>(input, bias, output, num_input,
                                                          hidden_dim);
  if (common::RunConfig::option_samback_cuda_launch_blocking) {
    CUDA_CALL(cudaStreamSynchronize(op->_model->stream));
  }
}

template <typename T>
__host__ void backward_task(Bias *op, T *input_grad, T *bias_grad, const T *output_grad,
                            const T* ones, const size_t num_input, const size_t hidden_dim) {
  if (input_grad != nullptr) {
    CUDA_CALL(cudaMemcpyAsync(input_grad, output_grad, num_input * hidden_dim * sizeof(T),
                    cudaMemcpyDeviceToDevice, op->_model->stream));
    if (common::RunConfig::option_samback_cuda_launch_blocking) {
      CUDA_CALL(cudaStreamSynchronize(op->_model->stream));
    }
  }
  CHECK(bias_grad != nullptr);

  TrainType alpha = 1, beta = 0;
  CUBLAS_CALL(cublasSgemv_v2(op->_model->blas, cublasOperation_t::CUBLAS_OP_N,
                          hidden_dim, num_input, &alpha, output_grad, hidden_dim,
                          ones, 0, // x, is 1.
                          &beta, bias_grad, 1));

  if (common::RunConfig::option_samback_cuda_launch_blocking) {
    CUDA_CALL(cudaStreamSynchronize(op->_model->stream));
  }
}

} // namespace

Bias::Bias(Model *model, const GradTensorPtr input, IdType dim) : GnnOp(model, input) {
  _bias = GradTensor::Empty(to_data_type<TrainType>(), {dim}, model->ctx, "bias", true);
  ZerosInitializer init;
  init.init(model, _bias->data());
  output = GradTensor::Null(true);
}

void Bias::forward() {
  size_t num_input = inputs[0]->data()->Shape()[0];
  size_t num_hidden = inputs[0]->data()->Shape()[1];
  const TrainType *input_ptr = tensor_cast_const<TrainType>(inputs[0]->data());
  const TrainType *bias_ptr = tensor_cast_const<TrainType>(_bias->data());
  TrainType *output_ptr = tensor_cast<TrainType>(output->data());
  forward_task(this, input_ptr, bias_ptr, output_ptr, num_input, num_hidden);
}

void Bias::backward() {
  if (output->RequireGrad() == false)
    return;
  size_t num_input = inputs[0]->data()->Shape()[0];
  size_t num_hidden = inputs[0]->data()->Shape()[1];
  TrainType *input_grad = nullptr, *bias_grad = tensor_cast<TrainType>(_bias->grad());
  if (inputs[0]->RequireGrad())
    input_grad = tensor_cast<TrainType>(inputs[0]->grad());
  const TrainType *output_grad = tensor_cast_const<TrainType>(output->grad());
  const TrainType *ones = tensor_cast_const<TrainType>(_model->ones);
  backward_task(this, input_grad, bias_grad, output_grad, ones, num_input, num_hidden);
  PrintTensor(_bias->grad(), "bias's grad: ");
}

void Bias::prepare() {
  TaskPtr cur_task = _model->cur_task;
  output->Resize(inputs[0]->Type(), inputs[0]->Shape(), _model->ctx, "");
  if (!_model->ones->Defined() || _model->ones->Shape()[0] < 1) {
    _model->ones->ChangeShape(to_data_type<TrainType>(), {1}, _model->ctx, "_model->ones");
    ValInitializer *init = new ValInitializer(1.0);
    init->init(_model, _model->ones);
    delete init;
  }
}

bool Bias::generate_grad() const { return true; }
bool Bias::accumulate_grad() const { return false; }
std::string Bias::name() const { return "Bias"; }

} // namespace sam_backend
} // namespace samgraph