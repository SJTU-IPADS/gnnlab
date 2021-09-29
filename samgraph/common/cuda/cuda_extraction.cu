#include "../common.h"
#include "../device.h"
#include "../logging.h"
#include "cuda_function.h"
#include "../run_config.h"

namespace samgraph {
namespace common {
namespace cuda {

namespace {

template <typename T>
__global__ void gpu_extract(void *output, const void *src, const IdType *index,
                            const size_t num_index, const size_t dim) {
  T *output_data = reinterpret_cast<T *>(output);
  const T *src_data = reinterpret_cast<const T *>(src);

  size_t i = blockIdx.x * blockDim.y + threadIdx.y;
  const size_t stride = blockDim.y * gridDim.x;

  while (i < num_index) {
    size_t col = threadIdx.x;
    const size_t src_idx = index[i];
    while (col < dim) {
      output_data[i * dim + col] = src_data[src_idx * dim + col];
      col += blockDim.x;
    }

    i += stride;
  }
}
template <typename T>
__global__ void gpu_mock_extract(
    void *output, const void *src, const IdType *index,
    const size_t num_index, const size_t dim, const size_t idx_mock_mask) {
  T *output_data = reinterpret_cast<T *>(output);
  const T *src_data = reinterpret_cast<const T *>(src);

  size_t i = blockIdx.x * blockDim.y + threadIdx.y;
  const size_t stride = blockDim.y * gridDim.x;

  while (i < num_index) {
    size_t col = threadIdx.x;
    const size_t src_idx = index[i] & idx_mock_mask;
    while (col < dim) {
      output_data[i * dim + col] = src_data[src_idx * dim + col];
      col += blockDim.x;
    }

    i += stride;
  }
}

}  // namespace

void GPUExtract(void *dst, const void *src, const IdType *index,
                const size_t num_index, const size_t dim, DataType dtype,
                Context ctx, StreamHandle stream, uint64_t task_key) {
  auto device = Device::Get(ctx);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  dim3 block(256, 1);
  while (static_cast<size_t>(block.x) >= 2 * dim) {
    block.x /= 2;
    block.y *= 2;
  }
  const dim3 grid(RoundUpDiv(num_index, static_cast<size_t>(block.y)));

  switch (dtype) {
    case kF32:
      gpu_extract<float>
          <<<grid, block, 0, cu_stream>>>(dst, src, index, num_index, dim);
      break;
    case kF64:
      gpu_extract<double>
          <<<grid, block, 0, cu_stream>>>(dst, src, index, num_index, dim);
      break;
    case kF16:
      gpu_extract<short>
          <<<grid, block, 0, cu_stream>>>(dst, src, index, num_index, dim);
      break;
    case kU8:
      gpu_extract<uint8_t>
          <<<grid, block, 0, cu_stream>>>(dst, src, index, num_index, dim);
      break;
    case kI32:
      gpu_extract<int32_t>
          <<<grid, block, 0, cu_stream>>>(dst, src, index, num_index, dim);
      break;
    case kI64:
      gpu_extract<int64_t>
          <<<grid, block, 0, cu_stream>>>(dst, src, index, num_index, dim);
      break;
    default:
      CHECK(0);
  }

  device->StreamSync(ctx, stream);
}

void GPUMockExtract(void *dst, const void *src, const IdType *index,
                const size_t num_index, const size_t dim, DataType dtype,
                Context ctx, StreamHandle stream, uint64_t task_key) {
  auto device = Device::Get(ctx);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  dim3 block(256, 1);
  while (static_cast<size_t>(block.x) >= 2 * dim) {
    block.x /= 2;
    block.y *= 2;
  }
  const dim3 grid(RoundUpDiv(num_index, static_cast<size_t>(block.y)));
  size_t idx_mock_mask = (1ull << RunConfig::option_empty_feat) - 1;

  switch (dtype) {
    case kF32:
      gpu_mock_extract<float>
          <<<grid, block, 0, cu_stream>>>(dst, src, index, num_index, dim, idx_mock_mask);
      break;
    case kF64:
      gpu_mock_extract<double>
          <<<grid, block, 0, cu_stream>>>(dst, src, index, num_index, dim, idx_mock_mask);
      break;
    case kF16:
      gpu_mock_extract<short>
          <<<grid, block, 0, cu_stream>>>(dst, src, index, num_index, dim, idx_mock_mask);
      break;
    case kU8:
      gpu_mock_extract<uint8_t>
          <<<grid, block, 0, cu_stream>>>(dst, src, index, num_index, dim, idx_mock_mask);
      break;
    case kI32:
      gpu_mock_extract<int32_t>
          <<<grid, block, 0, cu_stream>>>(dst, src, index, num_index, dim, idx_mock_mask);
      break;
    case kI64:
      gpu_mock_extract<int64_t>
          <<<grid, block, 0, cu_stream>>>(dst, src, index, num_index, dim, idx_mock_mask);
      break;
    default:
      CHECK(0);
  }

  device->StreamSync(ctx, stream);
}
}  // namespace cuda
}  // namespace common
}  // namespace samgraph