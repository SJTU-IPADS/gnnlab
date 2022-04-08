#include "cudnn.h"
#include "model.h"
#include "utils.h"
#include <cassert>
#include <iostream>

namespace samgraph {
namespace sam_backend {

// logits , label, output(logits_grad)
// output = softmax(input); grad_input = output - (0,...,0,1,0,...,0)
//                                                         ^ correct label
namespace {

struct PerfMetrics {
  float trainLoss;
  // IdType all;
  IdType correct;
};

// in samgraph, label is not a tensor, rather than a per-vertex scalar.

// https://zhuanlan.zhihu.com/p/105722023
// average by vertex: required when batch training
template <typename T>
__global__ void softmax_backward(const Id64Type *labels, const T* output, T *input_grad, int hiddenDim,
                                 IdType numVertices) {
  size_t v = blockIdx.x * blockDim.y + threadIdx.y;
  const size_t stride = blockDim.y * gridDim.x;

  while (v < numVertices) {
    size_t col = threadIdx.x;
    int minus_factor = (col == labels[v]) ? 1 : 0;
    while (col < hiddenDim) {
      size_t local_idx = v * hiddenDim + col;
      input_grad[local_idx] = output[local_idx] - minus_factor;
      input_grad[local_idx] /= numVertices;
      col += blockDim.x;
    }

    v += stride;
  }
}
// we ensure all passed in vertices is all train or val or test
template <typename T, size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void calc_loss(const T *output, const Id64Type *labels, const VertexType vt, float *loss,
                          IdType *num_correct, const int hiddenDim, const IdType numVertices) {
  assert(BLOCK_SIZE == blockDim.x);

  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  float local_loss = 0;
#pragma unroll
  for (IdType v = threadIdx.x + block_start; v < block_end; v += BLOCK_SIZE) {
    if (v >= numVertices) continue;
    T maxVal = 0.0f;
    Id64Type trueLabel = labels[v], myLabel = Constant::kEmptyLabel;
    for (Id64Type i = 0; i < hiddenDim; i++) {
      if (output[v * hiddenDim + i] <= maxVal)
        continue;
      maxVal = output[v * hiddenDim + i];
      myLabel = i;
    }
    if (trueLabel == myLabel) {
      atomicAdd(num_correct, 1);
    }
    if (vt == kVertexTrain) {
      local_loss += -log(output[v * hiddenDim + trueLabel]);
    }
  }
  atomicAdd(loss, local_loss);
}

/**
 * @brief calculate loss. output is used to store temporal results. it's not needed in backward
 * progress
 */
template <typename T>
__host__ void forward_task(SoftmaxCrossEntropy *op, int hiddenDim, const T *input,
                           const Id64Type *labels, T *output, IdType num_vertices, VertexType vt) {
  cudnnHandle_t dnn = op->_model->dnn;

  cudnnTensorDescriptor_t &inputDesc = op->tensor_desc;
  int dims[] = {(int)(num_vertices), hiddenDim, 1, 1};
  int strides[] = {dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1};

  CUDNN_CALL(cudnnSetTensorNdDescriptor(inputDesc, CUDNN_DATA_FLOAT, 4, dims, strides));
  float alpha = 1.0f, beta = 0.0f;
  CUDNN_CALL(cudnnSoftmaxForward(dnn, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha,
                                 inputDesc, input, &beta, inputDesc, output));
  if (common::RunConfig::option_samback_cuda_launch_blocking) {
    CUDA_CALL(cudaStreamSynchronize(op->_model->stream));
  }
}

// input is not used. grad = output - label
// input_grad can be alias of output
template <typename T>
__host__ void backward_task_ver2(SoftmaxCrossEntropy *op, int hiddenDim, const T *output,
                                 const Id64Type *labels, T *input_grad, size_t num_vertices,
                                 VertexType vt) {
  CHECK(vt == kVertexTrain);
  dim3 block(256, 1);
  while (static_cast<size_t>(block.x) >= 2 * hiddenDim) {
    block.x /= 2;
    block.y *= 2;
  }
  const dim3 grid(RoundUpDiv(num_vertices, static_cast<size_t>(block.y)));
  softmax_backward<<<grid, block, 0, op->_model->stream>>>(labels, output, input_grad, hiddenDim,
                                                           num_vertices);
  if (common::RunConfig::option_samback_cuda_launch_blocking) {
    CUDA_CALL(cudaStreamSynchronize(op->_model->stream));
  }
}

} // namespace

SoftmaxCrossEntropy::SoftmaxCrossEntropy(Model *model, const GradTensorPtr input)
    : GnnOp(model, input) {
  // dead hack: letting output to point at input's grad.
  // this reduces calculation during backward
  // output = GradTensor::FromTensor(input->grad());
  output = GradTensor::Null();
  CUDNN_CALL(cudnnCreateTensorDescriptor(&tensor_desc));
}
SoftmaxCrossEntropy::~SoftmaxCrossEntropy() {
  CUDNN_CALL(cudnnDestroyTensorDescriptor(tensor_desc));
}

void SoftmaxCrossEntropy::forward() {
  if (check_nan_exist(inputs[0]->data()))
    std::cout << "softmax founds nan\n";
  IdType num_vertices = inputs[0]->Shape()[0];
  IdType hiddem_dim = inputs[0]->Shape()[1];
  const TrainType *input_ptr = tensor_cast_const<TrainType>(inputs[0]->data());
  const Id64Type *labels = tensor_cast_const<Id64Type>(_model->cur_task->output_label);
  TrainType *output_ptr = tensor_cast<TrainType>(output->data());
  forward_task(this, hiddem_dim, input_ptr, labels, output_ptr, num_vertices, kVertexTrain);
  if (check_nan_exist(output->data()))
    std::cout << "softmax founds nan in output\n";
}
void SoftmaxCrossEntropy::backward() {
  IdType num_vertices = inputs[0]->Shape()[0];
  IdType hiddem_dim = inputs[0]->Shape()[1];
  const Id64Type *labels = tensor_cast_const<Id64Type>(_model->cur_task->output_label);
  const TrainType *output_ptr = tensor_cast<TrainType>(output->data());
  TrainType *input_grad = tensor_cast<TrainType>(inputs[0]->grad());
  // backward_task(this, hiddem_dim, output_ptr, labels, input_grad, num_vertices, kVertexTrain);
  backward_task_ver2(this, hiddem_dim, output_ptr, labels, input_grad, num_vertices, kVertexTrain);
}

void SoftmaxCrossEntropy::prepare() {
  TaskPtr cur_task = _model->cur_task;

  output->Resize(inputs[0]->Type(), inputs[0]->Shape(), _model->ctx, "");
}

bool SoftmaxCrossEntropy::generate_grad()   const {return true;}
bool SoftmaxCrossEntropy::accumulate_grad() const {return false;}
std::string SoftmaxCrossEntropy::name() const {return "softmax";}

void SoftmaxCrossEntropy::loss(float &loss, float &accuracy) {
  IdType num_vertices = inputs[0]->Shape()[0];
  IdType hiddem_dim = inputs[0]->Shape()[1];
  const Id64Type *labels = tensor_cast_const<Id64Type>(_model->cur_task->output_label);
  TrainType *output_ptr = tensor_cast<TrainType>(output->data());

  // Calculate loss
  PerfMetrics *perf_gpu;
  PerfMetrics perf_cpu;
  perf_cpu.trainLoss = 0.0f;
  perf_cpu.correct = 0;

  CUDA_CALL(cudaMalloc(&perf_gpu, sizeof(PerfMetrics)));
  CUDA_CALL(cudaMemcpyAsync(perf_gpu, &perf_cpu, sizeof(PerfMetrics), cudaMemcpyHostToDevice,
                            _model->stream));

  const size_t num_tiles = RoundUpDiv(static_cast<size_t>(num_vertices), Constant::kCudaTileSize);
  const dim3 grid(num_tiles);
  const dim3 block(Constant::kCudaBlockSize);
  calc_loss<TrainType, Constant::kCudaBlockSize, Constant::kCudaTileSize><<<grid, block, 0, _model->stream>>>(
      output_ptr, labels, kVertexTrain, &perf_gpu->trainLoss, &perf_gpu->correct, hiddem_dim, num_vertices);
  CUDA_CALL(cudaMemcpyAsync(&perf_cpu, perf_gpu, sizeof(PerfMetrics), cudaMemcpyDeviceToHost,
                            _model->stream));
  CUDA_CALL(cudaStreamSynchronize(_model->stream));
  loss = perf_cpu.trainLoss / num_vertices;
  accuracy = perf_cpu.correct * 100.0f / num_vertices;
}

} // namespace sam_backend
} // namespace samgraph