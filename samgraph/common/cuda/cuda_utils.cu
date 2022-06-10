#include "../constant.h"
#include <cuda_runtime.h>
#include <cassert>

namespace samgraph {
namespace common {
namespace cuda {

#define SAM_1D_GRID_INIT(num_input) \
  const size_t num_tiles = RoundUpDiv((size_t)num_input, Constant::kCudaTileSize); \
  const dim3 grid(num_tiles);                                              \
  const dim3 block(Constant::kCudaBlockSize);                              \

#define SAM_1D_GRID_FOR(loop_iter, num_input) \
  assert(BLOCK_SIZE == blockDim.x);                       \
  const size_t block_start = TILE_SIZE * blockIdx.x;      \
  const size_t block_end = min(TILE_SIZE * (blockIdx.x + 1), num_input);  \
  for (size_t loop_iter = threadIdx.x + block_start; loop_iter < block_end; loop_iter += BLOCK_SIZE) \

namespace {

template <typename T, size_t BLOCK_SIZE=Constant::kCudaBlockSize, size_t TILE_SIZE=Constant::kCudaTileSize>
__global__ void arrange_array(T* array, size_t array_len, T begin, T step) {
  SAM_1D_GRID_FOR(i, array_len) {
    array[i] = begin + i * step;
  }
}

}

template <typename T>
void ArrangeArray(T* array, size_t array_len, T begin, T step, StreamHandle stream) {
  SAM_1D_GRID_INIT(array_len);
  auto cu_stream = static_cast<cudaStream_t>(stream);
  arrange_array<<<grid, block, 0, cu_stream>>>(array, array_len, begin, step);
}

template void ArrangeArray<int>(int*, size_t, int, int, StreamHandle);

namespace {
__device__ IdType _UpperBound(const IdType *A, int64_t n, IdType x) {
  IdType l = 0, r = n, m = 0;
  while (l < r) {
    m = l + (r-l)/2;
    if (x >= A[m]) {
      l = m+1;
    } else {
      r = m;
    }
  }
  return l;
}

template <size_t BLOCK_SIZE=Constant::kCudaBlockSize, size_t TILE_SIZE=Constant::kCudaTileSize>
__global__ void get_row_from_eid (const IdType * indptr, size_t n_row, const IdType *eid_list, size_t num_eid, IdType * output_row) {
  SAM_1D_GRID_FOR(i, num_eid) {
    output_row[i] = _UpperBound(indptr, n_row, eid_list[i]) - 1;
  }
}
}

void GPUGetRowFromEid(const IdType * indptr, size_t n_row, const IdType *eid_list, size_t num_eid, IdType * output_row, StreamHandle stream) {
  SAM_1D_GRID_INIT(num_eid);
  auto cu_stream = static_cast<cudaStream_t>(stream);
  get_row_from_eid<><<<grid, block, 0, cu_stream>>>(indptr, n_row, eid_list, num_eid, output_row);
}

}
}
}