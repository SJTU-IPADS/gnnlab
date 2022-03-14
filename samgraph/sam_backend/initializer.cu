#include "common.h"
#include "model.h"

#include <cassert>
#include <ctime>
#include <curand.h>

namespace samgraph {
namespace sam_backend {

namespace {

template <typename T, size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void scale_kernel(int count, T a, T b, T *ptr) {
  assert(BLOCK_SIZE == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);
  for (int i = threadIdx.x + block_start; i < block_end; i += BLOCK_SIZE) {
    if (i >= count) continue;
    ptr[i] = (b - a) * ptr[i] + a;
  }
}
template <typename T, size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void assign_kernel(int count, T value, T *ptr) {
  assert(BLOCK_SIZE == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);
  for (int i = threadIdx.x + block_start; i < block_end; i += BLOCK_SIZE) {
    if (i >= count) continue;
    ptr[i] = value;
  }
}

template <typename T>
void glorot_uniform_init(T *ptr, size_t input_dim, size_t output_dim, unsigned long long seed,
                         cudaStream_t stream) {
  float scale = sqrt(6.0 / (input_dim + output_dim));
  // printf("scale = %.4lf\n", scale);
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  // TODO: change to random seed before releasing
  // fprintf(stderr, "seed = %d\n", seed);
  curandSetPseudoRandomGeneratorSeed(gen, seed);
  CURAND_CALL(curandGenerateUniform(gen, ptr, input_dim * output_dim));
  dim3 grid(RoundUpDiv(input_dim * output_dim, Constant::kCudaTileSize));
  dim3 block(Constant::kCudaBlockSize);
  scale_kernel<T, Constant::kCudaBlockSize, Constant::kCudaTileSize>
              <<<grid, block, 0, stream>>>(input_dim * output_dim, -scale, scale, ptr);
  CUDA_CALL(cudaStreamSynchronize(stream));
  curandDestroyGenerator(gen);
}

template <typename T>
void zero_init(T *ptr, size_t count, cudaStream_t stream) {
  dim3 grid(RoundUpDiv(count, Constant::kCudaTileSize));
  dim3 block(Constant::kCudaBlockSize);
  assign_kernel<T, Constant::kCudaBlockSize, Constant::kCudaTileSize>
               <<<grid, block, 0, stream>>>(count, (T)0, ptr);
  CUDA_CALL(cudaStreamSynchronize(stream));
}

template <typename T>
void val_init(T *ptr, size_t count, T val, cudaStream_t stream) {
  dim3 grid(RoundUpDiv(count, Constant::kCudaTileSize));
  dim3 block(Constant::kCudaBlockSize);
  assign_kernel<T, Constant::kCudaBlockSize, Constant::kCudaTileSize>
               <<<grid, block, 0, stream>>>(count, val, ptr);
  CUDA_CALL(cudaStreamSynchronize(stream));
}
} // namespace

Initializer::Initializer(void) {}

Initializer::~Initializer(void) {}

GlorotUniform::GlorotUniform(void) : Initializer() {}

GlorotUniform::~GlorotUniform(void) {}

void GlorotUniform::init(const Model *model, const TensorPtr p) {
  Context ctx = model->ctx;
  CUDA_CALL(cudaSetDevice(ctx.device_id));
  CHECK(p->Shape().size() == 2);
  unsigned long long seed = std::chrono::system_clock::now().time_since_epoch().count();
  glorot_uniform_init(tensor_cast<TrainType>(p), p->Shape()[1], p->Shape()[0], seed, model->stream);
}

ZerosInitializer::ZerosInitializer(void) : Initializer() {}

ZerosInitializer::~ZerosInitializer(void) {}

void ZerosInitializer::init(const Model *model, const TensorPtr p) {
  // Context ctx = model->ctx;
  size_t count = p->NumItem();
  switch (p->Type()) {
  case common::DataType::kF32:
    zero_init(tensor_cast<float>(p), count, model->stream);
    break;
  case common::DataType::kI32:
    zero_init(tensor_cast<uint32_t>(p), count, model->stream);
    break;
  case common::DataType::kI64:
    zero_init(tensor_cast<uint64_t>(p), count, model->stream);
    break;
  default: CHECK(false);
  }
}

ValInitializer::ValInitializer(TrainType val) : Initializer() { this->val = val; }

ValInitializer::~ValInitializer(void) {}

void ValInitializer::init(const Model *model, const TensorPtr p) {
  // Context ctx = model->ctx;
  size_t count = p->NumItem();
  val_init(tensor_cast<TrainType>(p), count, this->val, model->stream);
}

} // namespace sam_backend
} // namespace samgraph