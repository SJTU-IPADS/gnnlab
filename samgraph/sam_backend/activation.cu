#include "cudnn.h"
#include "utils.h"
#include "model.h"
#include <iostream>

namespace samgraph {
namespace sam_backend {

// input's grad is overrided rather than accumulated.
namespace {

template <typename T>
__host__ void forward_task(
    Activation* op, const T *input, T *output,
    const size_t len, const size_t hidden, cudaStream_t stream) {
  cudnnTensorDescriptor_t& inTensor = op->tensorDesc;
  cudnnActivationDescriptor_t &actiDesc = op->actiDesc;
  cudnnHandle_t &dnn = op->_model->dnn;
  int dims[] = {(int)len, (int)hidden, 1};
  int strides[] = {dims[1] * dims[2], dims[2], 1};
  float alpha = 1.0f, beta = 0.0f;
  CUDNN_CALL(cudnnSetTensorNdDescriptor(inTensor, to_cudnn_data_type<T>(),
                                        3, dims, strides));
  CUDNN_CALL(cudnnActivationForward(dnn, actiDesc,
                                    &alpha, inTensor, input,
                                    &beta, inTensor, output));
  if (common::RunConfig::option_samback_cuda_launch_blocking) {
    CUDA_CALL(cudaStreamSynchronize(stream));
  }
}

template <typename T>
__host__ void backward_task(
    Activation* op,
    const T *output_grad, const T *output, const T *input, T *input_grad,
    const size_t len, const size_t hidden, cudaStream_t stream) {
  float alpha = 1.0f, beta = 0.0f;
  cudnnTensorDescriptor_t &outTensor = op->tensorDesc;
  cudnnActivationDescriptor_t &actiDesc = op->actiDesc;
  cudnnHandle_t &dnn = op->_model->dnn;
  CUDNN_CALL(cudnnActivationBackward(dnn, actiDesc,
                                     &alpha, outTensor, output,
                                     outTensor, output_grad,
                                     outTensor, input,
                                     &beta, outTensor, input_grad));
  if (common::RunConfig::option_samback_cuda_launch_blocking) {
    CUDA_CALL(cudaStreamSynchronize(stream));
  }
}

} // namespace

Activation::Activation(Model* model,
                          const GradTensorPtr input,
                          ActiMode _actiMode)
    : GnnOp(model, input), actiMode(_actiMode) {
  output = GradTensor::Null(input->RequireGrad());
  CUDNN_CALL(cudnnCreateActivationDescriptor(&actiDesc));
  switch (actiMode) {
  case KAcModeRelu:
    CUDNN_CALL(cudnnSetActivationDescriptor(
        actiDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0));
    break;
  case KAcModeSigmoid:
    CUDNN_CALL(cudnnSetActivationDescriptor(
        actiDesc, CUDNN_ACTIVATION_SIGMOID, CUDNN_PROPAGATE_NAN, 0.0));
    break;
  default:
    CHECK(false);
  }
  CUDNN_CALL(cudnnCreateTensorDescriptor(&tensorDesc));
}
Activation::~Activation() {
  CUDNN_CALL(cudnnDestroyActivationDescriptor(actiDesc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(tensorDesc));
}

void Activation::forward() {
  if (check_nan_exist(inputs[0]->data())) std::cout << "activation founds nan\n";
  forward_task(this,
               tensor_cast<TrainType>(this->inputs[0]->data()),
               tensor_cast<TrainType>(this->output->data()),
               this->inputs[0]->Shape()[0],
               this->inputs[0]->Shape()[1], _model->stream);
}

void Activation::backward() {
  if (inputs[0]->RequireGrad() == false) return;
  backward_task(this,
                tensor_cast<TrainType>(this->output->grad()),
                tensor_cast<TrainType>(this->output->data()),
                tensor_cast<TrainType>(this->inputs[0]->data()),
                tensor_cast<TrainType>(this->inputs[0]->grad()),
                this->inputs[0]->Shape()[0],
                this->inputs[0]->Shape()[1], _model->stream);
}

void Activation::prepare() {
  output->Resize(inputs[0]->Type(), inputs[0]->Shape(), _model->ctx, "");
}

bool Activation::generate_grad()   const {return true;}
bool Activation::accumulate_grad() const {return false;}
std::string Activation::name() const {return "activation";}

} // namespace sam_backend
} // namespace samgraph