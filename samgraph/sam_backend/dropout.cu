#include "model.h"
#include "utils.h"
#include <iostream>

namespace samgraph {
namespace sam_backend {

namespace {

__host__ void init_task(Dropout *dropout_ptr) {
  cudnnDropoutDescriptor_t &dropoutDesc = dropout_ptr->dropout_desc;
  cudnnTensorDescriptor_t &outputDesc = dropout_ptr->tensor_desc;
  cudnnHandle_t dnn = dropout_ptr->_model->dnn;
  Context ctx = dropout_ptr->_model->ctx;
  GPUDevice *device = static_cast<GPUDevice *>(GPUDevice::Get(ctx));

  CUDNN_CALL(cudnnDropoutGetStatesSize(dnn, &(dropout_ptr->dropout_states_size)));
  dropout_ptr->dropout_states = device->AllocDataSpace(ctx, dropout_ptr->dropout_states_size);

  CUDNN_CALL(cudnnCreateDropoutDescriptor(&dropoutDesc));
  CUDNN_CALL(cudnnSetDropoutDescriptor(dropoutDesc, dnn, dropout_ptr->rate,
                                       dropout_ptr->dropout_states,
                                       dropout_ptr->dropout_states_size, dropout_ptr->seed));
  CUDNN_CALL(cudnnCreateTensorDescriptor(&outputDesc));
}

template <typename T>
__host__ void forward_task(Dropout *dropout_ptr, const T *input, T *output, const size_t len,
                           const size_t hidden, cudaStream_t stream) {
  cudnnTensorDescriptor_t &outputDesc = dropout_ptr->tensor_desc;
  int dims[] = {(int)len, (int)hidden, 1};
  int strides[] = {dims[1] * dims[2], dims[2], 1};
  CUDNN_CALL(cudnnSetTensorNdDescriptor(outputDesc, CUDNN_DATA_FLOAT, 3, dims, strides));
  cudnnHandle_t dnn = dropout_ptr->_model->dnn;
  void * &space = dropout_ptr->space;
  size_t &space_size = dropout_ptr->space_size;
  Context ctx = dropout_ptr->_model->ctx;
  cudnnDropoutDescriptor_t dropoutDesc = dropout_ptr->dropout_desc;
  CUDNN_CALL(cudnnDropoutGetReserveSpaceSize(outputDesc, &space_size));
  if (dropout_ptr->actual_space_size < space_size) {
    if (space) {
      common::Device::Get(ctx)->FreeWorkspace(ctx, space);
    }
    space = common::Device::Get(ctx)->AllocWorkspace(ctx, space_size);
    dropout_ptr->actual_space_size = common::Device::Get(ctx)->WorkspaceActualSize(ctx, space);
  }

  CUDNN_CALL(cudnnDropoutForward(dnn, dropoutDesc, outputDesc, input, outputDesc, output, space,
                                 space_size));
  if (common::RunConfig::option_samback_cuda_launch_blocking) {
    CUDA_CALL(cudaStreamSynchronize(stream));
  }
}

template <typename T>
__host__ void backward_task(Dropout *dropout_ptr, const T *output_grad, T *input_grad,
                            const size_t len, const size_t hidden, cudaStream_t stream) {
  cudnnTensorDescriptor_t &outputDesc = dropout_ptr->tensor_desc;
  cudnnHandle_t dnn = dropout_ptr->_model->dnn;
  void *&space = dropout_ptr->space;
  size_t &space_size = dropout_ptr->space_size;
  Context ctx = dropout_ptr->_model->ctx;
  cudnnDropoutDescriptor_t dropoutDesc = dropout_ptr->dropout_desc;
  CHECK(dropout_ptr->space != nullptr);
  CUDNN_CALL(cudnnDropoutBackward(dnn, dropoutDesc, outputDesc, output_grad, outputDesc, input_grad,
                                  space, space_size));
  if (common::RunConfig::option_samback_cuda_launch_blocking) {
    CUDA_CALL(cudaStreamSynchronize(stream));
  }
}

template <typename T>
__host__ void infer_task(Dropout *dropout_ptr, const T *input, T *output, const size_t len,
                         const size_t hidden, cudaStream_t stream) {

  Context ctx = dropout_ptr->_model->ctx;
  GPUDevice *device = static_cast<GPUDevice *>(GPUDevice::Get(ctx));

  device->CopyDataFromTo(input, 0, output, 0, len * hidden * sizeof(T), dropout_ptr->_model->ctx,
                         dropout_ptr->_model->ctx, stream);

  CUDA_CALL(cudaStreamSynchronize(stream));
}
} // namespace

Dropout::Dropout(Model *model, const GradTensorPtr input, float rate, int seed)
    : GnnOp(model, input) {
  this->rate = rate;
  this->seed = seed;

  init_task(this);
  output = GradTensor::Null(input->RequireGrad());
}

Dropout::~Dropout() {
  CUDNN_CALL(cudnnDestroyDropoutDescriptor(dropout_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(tensor_desc));
  Device::Get(_model->ctx)->FreeDataSpace(_model->ctx, dropout_states);
  if (space) {
    Device::Get(_model->ctx)->FreeWorkspace(_model->ctx, space);
  }
}

void Dropout::forward() {
  if (check_nan_exist(inputs[0]->data())) std::cout << "dropout founds nan\n";
  if (_model->mode == KModeTrain) {
    forward_task(this, tensor_cast<TrainType>(this->inputs[0]->data()),
                 tensor_cast<TrainType>(this->output->data()), this->inputs[0]->Shape()[0],
                 this->inputs[0]->Shape()[1], this->_model->stream);
  } else {
    infer_task(this, tensor_cast<TrainType>(this->inputs[0]->data()),
               tensor_cast<TrainType>(this->output->data()), this->inputs[0]->Shape()[0],
               this->inputs[0]->Shape()[1], this->_model->stream);
  }
}

void Dropout::backward() {
  CHECK(_model->mode == KModeTrain);
  if (_model->mode == KModeTrain) {
    if (this->inputs[0]->RequireGrad() == false) return;
    backward_task(this, tensor_cast<TrainType>(this->output->grad()),
                  tensor_cast<TrainType>(this->inputs[0]->grad()), this->inputs[0]->Shape()[0],
                  this->inputs[0]->Shape()[1], this->_model->stream);
  } else {
    infer_task(this, tensor_cast<TrainType>(this->output->grad()),
               tensor_cast<TrainType>(this->inputs[0]->grad()), this->inputs[0]->Shape()[0],
               this->inputs[0]->Shape()[1], this->_model->stream);
  }
}

void Dropout::prepare() {
  TaskPtr cur_task = _model->cur_task;

  output->Resize(inputs[0]->Type(), inputs[0]->Shape(), _model->ctx, "");
}

bool Dropout::generate_grad()   const {return true;}
bool Dropout::accumulate_grad() const {return false;}
std::string Dropout::name() const {return "dropout";}

} // namespace sam_backend
} // namespace samgraph