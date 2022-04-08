
#include "model.h"
#include <cub/cub.cuh>
#include <cusparse.h>

namespace samgraph {
namespace sam_backend {

namespace {

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void prepare_key(const IdType *low, const IdType *high, Id64Type *output, size_t size) {

  assert(BLOCK_SIZE == blockDim.x);

  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);
#pragma unroll
  for (size_t i = threadIdx.x + block_start; i < block_end; i += BLOCK_SIZE) {
    if (i < size) {
      output[i] = (((Id64Type)high[i]) << 32) | low[i];
    }
  }
}
template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void extract_key_val(IdType *low, IdType *high, const Id64Type *output, size_t size) {

  assert(BLOCK_SIZE == blockDim.x);

  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);
#pragma unroll
  for (size_t i = threadIdx.x + block_start; i < block_end; i += BLOCK_SIZE) {
    if (i < size) {
      low[i] = output[i];
      high[i] = output[i] >> 32;
    }
  }
}
} // namespace
/**
 * @brief sort a coo to be row-major
 */
void sort_coo(TensorPtr row, TensorPtr col, int m, int n, cusparseHandle_t sparse,
              cudaStream_t stream) {
  void *buffer = nullptr;
  size_t buffer_size = 0;
  // TensorPtr row_double_buf = Tensor::Empty(row->Type(), row->Shape(), row->Ctx(), "");
  // TensorPtr col_double_buf = Tensor::Empty(col->Type(), col->Shape(), col->Ctx(), "");
  // cub::DoubleBuffer<IdType> keys(tensor_cast<IdType>(row), tensor_cast<IdType>(row_double_buf));
  // cub::DoubleBuffer<IdType> vals(tensor_cast<IdType>(col), tensor_cast<IdType>(col_double_buf));
  // cub::DeviceRadixSort::SortPairs(buffer, buffer_size, keys, vals, row->Shape()[0], 0,
  //                                 sizeof(IdType) * 8, stream);
  // CUDA_CALL(cudaStreamSynchronize(stream));

  TensorPtr keys_double_buf1 =
      Tensor::Empty(common::kI64, row->Shape(), row->Ctx(), "sort_coo.buffer");
  TensorPtr keys_double_buf2 =
      Tensor::Empty(common::kI64, row->Shape(), row->Ctx(), "sort_coo.buffer");
  cub::DoubleBuffer<Id64Type> keys(tensor_cast<Id64Type>(keys_double_buf1),
                                   tensor_cast<Id64Type>(keys_double_buf2));

  const size_t num_tiles = RoundUpDiv(row->Shape()[0], Constant::kCudaTileSize);
  const dim3 grid(num_tiles);
  const dim3 block(Constant::kCudaBlockSize);

  cub::DeviceRadixSort::SortKeys(buffer, buffer_size, keys, row->Shape()[0], 0,
                                 sizeof(Id64Type) * 8, stream);
  CUDA_CALL(cudaStreamSynchronize(stream));
  buffer = Device::Get(row->Ctx())->AllocWorkspace(row->Ctx(), buffer_size);
  prepare_key<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid, block, 0, stream>>>(tensor_cast_const<IdType>(col), tensor_cast_const<IdType>(row),
                                   tensor_cast<Id64Type>(keys_double_buf1), row->Shape()[0]);
  cub::DeviceRadixSort::SortKeys(buffer, buffer_size, keys, row->Shape()[0], 0,
                                 sizeof(Id64Type) * 8, stream);
  extract_key_val<Constant::kCudaBlockSize, Constant::kCudaTileSize><<<grid, block, 0, stream>>>(
      tensor_cast<IdType>(col), tensor_cast<IdType>(row), keys.Current(), row->Shape()[0]);
  CUDA_CALL(cudaStreamSynchronize(stream));

  // cusparseXcoosort_bufferSizeExt(sparse, m, n, row->Shape()[0], (int *)tensor_cast<IdType>(row),
  //                                (int *)tensor_cast<IdType>(col), &buffer_size);
  // buffer = Device::Get(row->Ctx())->AllocWorkspace(row->Ctx(), buffer_size);
  // cusparseXcoosortByRow(sparse, m, n, row->Shape()[0], (int *)tensor_cast<IdType>(row),
  //                       (int *)tensor_cast<IdType>(col), nullptr, buffer);
  // CUDA_CALL(cudaStreamSynchronize(stream));

  Device::Get(row->Ctx())->FreeWorkspace(row->Ctx(), buffer);
}

namespace {
__device__ bool is_nan(float v) { return *((int *)&v) == 0x7fffffff; }
template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void check_nan_kernel(float *ptr, size_t num, size_t hidden, unsigned long long* rst) {
  assert(BLOCK_SIZE == blockDim.x);

  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);
#pragma unroll
  for (size_t i = threadIdx.x + block_start; i < block_end; i += BLOCK_SIZE) {
    unsigned long long nan_count = 0;
    if (i < num) {
      for (size_t h = 0; h < hidden; h++) {
        if (is_nan(ptr[i * hidden + h]))
          nan_count++;
      }
      if (nan_count > 0) {
        atomicAdd(rst, nan_count);
      }
    }
  }
}

} // namespace

bool check_nan_exist(TensorPtr ptr) {
  // const size_t num_tiles = RoundUpDiv(ptr->Shape()[0], Constant::kCudaTileSize);
  // const dim3 grid(num_tiles);
  // const dim3 block(Constant::kCudaBlockSize);
  // TensorPtr cpu_bool = Tensor::Empty(common::kI64, {1}, common::CPU(), "");
  // tensor_cast<uint64_t>(cpu_bool)[0] = 0;
  // TensorPtr gpu_bool = Tensor::CopyTo(cpu_bool, ptr->Ctx());
  // check_nan_kernel<Constant::kCudaBlockSize, Constant::kCudaTileSize>
  //     <<<grid, block>>>(tensor_cast<float>(ptr), ptr->Shape()[0], ptr->Shape()[1], (unsigned long long *)tensor_cast<uint64_t>(gpu_bool));
  // cpu_bool = Tensor::CopyTo(gpu_bool, common::CPU());
  // return tensor_cast<uint64_t>(cpu_bool)[0] > 0;
  return false;
}
} // namespace sam_backend
} // namespace samgraph