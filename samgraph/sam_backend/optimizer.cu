#include "../common/logging.h"
#include "common.h"
#include "constants.h"
#include "model.h"
#include "optimizer.h"
#include <cassert>
// #include "cuda_helper.h"

namespace samgraph {
namespace sam_backend {

namespace {
// template <typename T>
// __global__ void add_kernel(int count, T scale, const T *src, T *dst) {
//   for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
//     dst[i] += src[i] * scale;
//   }
// }

// template <typename T>
// __global__ void scale_kernel(int count, T a, T b, T *ptr) {
//   for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
//     ptr[i] = (b - a) * ptr[i] + a;
//   }
// }

template <typename T, size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void adam_update(T *w, T *m, T *v, const T *w_grad, const T lr, const T beta1,
                            const T beta2, const T stepped_beta1, const T stepped_beta2,
                            const T weight_decay, const T epsilon, const size_t count) {
  // Reference for weight decay
  // https://www.fast.ai/2018/07/02/adam-weight-decay/
  assert(BLOCK_SIZE == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);
#pragma unroll
  for (int i = threadIdx.x + block_start; i < block_end; i += BLOCK_SIZE) {
    if (i >= count) continue;
    T gt = w_grad[i] + weight_decay * w[i];
    T mt = beta1 * m[i] + (1 - beta1) * gt;
    T vt = beta2 * v[i] + (1 - beta2) * gt * gt;
    m[i] = mt;
    v[i] = vt;
    T denom = sqrt(vt) / sqrt(1 - stepped_beta2) + epsilon;
    T step_size = lr / (1 - stepped_beta1);
    w[i] -= mt * step_size / denom;
  }
}

template <typename T>
__host__ void adam_update_task(T *w, T *m, T *v, const T *w_grad, const T lr, const T beta1,
                               const T beta2, const T stepped_beta1, const T stepped_beta2,
                               const T weight_decay, const T epsilon, const size_t count,
                               StreamHandle stream) {
  const size_t num_tiles0 = RoundUpDiv(count, Constant::kCudaTileSize);
  dim3 grid(num_tiles0);
  dim3 block(Constant::kCudaBlockSize);

  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  adam_update<T, Constant::kCudaBlockSize, Constant::kCudaTileSize>
             <<<grid, block, 0, cuda_stream>>>(w, m, v, w_grad, lr, beta1, beta2, stepped_beta1,
                                               stepped_beta2, weight_decay, epsilon, count);
}

template <typename T, size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void naive_update(T *w, const T *w_grad, const T lr, const size_t count) {
  assert(BLOCK_SIZE == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);
#pragma unroll
  for (int i = threadIdx.x + block_start; i < block_end; i += BLOCK_SIZE) {
    if (i >= count) continue;
    w[i] -= lr * w_grad[i];
  }
}
template <typename T>
__host__ void naive_update_task(T *w, const T *w_grad, const T lr, const size_t count,
                                StreamHandle stream) {
  const size_t num_tiles0 = RoundUpDiv(count, Constant::kCudaTileSize);
  dim3 grid(num_tiles0);
  dim3 block(Constant::kCudaBlockSize);
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  naive_update<T, Constant::kCudaBlockSize, Constant::kCudaTileSize>
              <<<grid, block, 0, cuda_stream>>>(w, w_grad, lr, count);
  CUDA_CALL(cudaStreamSynchronize(cuda_stream));
}

} // namespace
Optimizer::Optimizer(const Model *model) : model(model) {}

void AdamOptimizer::update() {
  auto &parameters = model->parameters;
  for (int index = parameters.size() - 1; index >= 0; index--) {
    size_t count =
        std::accumulate(parameters[index]->data()->Shape().begin(),
                        parameters[index]->data()->Shape().end(), 1ul, std::multiplies<size_t>());
    TrainType *w_ptr = tensor_cast<TrainType>(parameters[index]->data());
    TrainType *m_ptr = tensor_cast<TrainType>(m[index]);
    TrainType *v_ptr = tensor_cast<TrainType>(v[index]);
    TrainType *w_grad_ptr = tensor_cast<TrainType>(parameters[index]->grad());
    adam_update_task(w_ptr, m_ptr, v_ptr, w_grad_ptr, _lr, _beta1, _beta2, _stepped_beta1,
                     _stepped_beta2, _weight_decay, _epsilon, count, model->stream);
  }
  CUDA_CALL(cudaStreamSynchronize(model->stream));
}

void AdamOptimizer::next(void) {
  _stepped_beta1 *= _beta1;
  _stepped_beta2 *= _beta2;
  // lr_t = _lr * sqrt(1 - beta2_t) / (1 - beta1_t);
  // fprintf(stderr, "lr = %.4lf alpha_t = %.4lf\n", alpha, alpha_t);
}

AdamOptimizer::AdamOptimizer(const Model *model, TrainType lr, TrainType weight_decay,
                             TrainType beta1, TrainType beta2, TrainType epsilon)
    : Optimizer(model), _lr(lr), _beta1(beta1), _beta2(beta2), _weight_decay(weight_decay),
      _epsilon(epsilon),
      _stepped_beta1(1.0f), _stepped_beta2(1.0f) {
  auto &parameters = model->parameters;
  Initializer *init = new ZerosInitializer();
  for (size_t i = 0; i < model->parameters.size(); i++) {
    v.push_back(Tensor::Empty(parameters[i]->Type(), parameters[i]->Shape(), model->ctx, ""));
    m.push_back(Tensor::Empty(parameters[i]->Type(), parameters[i]->Shape(), model->ctx, ""));
    init->init(model, v.back());
    init->init(model, m.back());
  }
  delete init;
}

NaiveOptimizer::NaiveOptimizer(const Model *model, TrainType lr) : Optimizer(model), _lr(lr) {}

void NaiveOptimizer::update() {
  auto &parameters = model->parameters;
  for (int index = parameters.size() - 1; index >= 0; index--) {
    size_t count =
        std::accumulate(parameters[index]->data()->Shape().begin(),
                        parameters[index]->data()->Shape().end(), 1ul, std::multiplies<size_t>());
    TrainType *w_ptr = tensor_cast<TrainType>(parameters[index]->data());
    TrainType *w_grad_ptr = tensor_cast<TrainType>(parameters[index]->grad());
    naive_update_task(w_ptr, w_grad_ptr, _lr, count, model->stream);
  }
}

void NaiveOptimizer::next(void) {}

} // namespace sam_backend
} // namespace samgraph