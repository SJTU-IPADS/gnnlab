#include "../constant.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <cassert>
#include <curand.h>
#include <curand_kernel.h>

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

#define SAM_1D_GRID_RND_IDX() \
  (blockDim.x * blockIdx.x + threadIdx.x)

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
  if (array_len == 0) return;
  SAM_1D_GRID_INIT(array_len);
  auto cu_stream = static_cast<cudaStream_t>(stream);
  arrange_array<<<grid, block, 0, cu_stream>>>(array, array_len, begin, step);
}

template void ArrangeArray<int>(int*, size_t, int, int, StreamHandle);
template void ArrangeArray<Id64Type>(Id64Type*, size_t, Id64Type, Id64Type, StreamHandle);
template void ArrangeArray<float>(float*, size_t, float, float, StreamHandle);

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
  if (n_row == 0) return;
  SAM_1D_GRID_INIT(num_eid);
  auto cu_stream = static_cast<cudaStream_t>(stream);
  get_row_from_eid<><<<grid, block, 0, cu_stream>>>(indptr, n_row, eid_list, num_eid, output_row);
}

namespace {
template <typename T, size_t BLOCK_SIZE=Constant::kCudaBlockSize, size_t TILE_SIZE=Constant::kCudaTileSize>
__global__ void fill_uniform(T* array, size_t array_len, T range_min, T range_max, curandState *random_states, size_t num_random_states) {
  size_t rnd_state_idx = SAM_1D_GRID_RND_IDX();
  assert(rnd_state_idx < num_random_states);
  curandState local_state = random_states[rnd_state_idx];
  SAM_1D_GRID_FOR(i, array_len) {
    array[i] = (curand(&local_state) % (range_max - range_min + 1)) + range_min;
  }
  random_states[rnd_state_idx] = local_state;
}
template <typename T, size_t BLOCK_SIZE=Constant::kCudaBlockSize, size_t TILE_SIZE=Constant::kCudaTileSize>
__global__ void fill_repeat(T* array, const T* src_array, size_t src_len, const size_t repeats) {
  SAM_1D_GRID_FOR(dst_idx, src_len * repeats) {
    const size_t src_idx = dst_idx / repeats;
    array[dst_idx] = src_array[src_idx];
  }
}
}

template<typename T>
void ArrayGenerator::byUniform(T* array, size_t array_len, T min, T max, curandState *random_states, size_t num_random_states, StreamHandle stream) {
  if (array_len == 0) return;
  SAM_1D_GRID_INIT(array_len);
  auto cu_stream = static_cast<cudaStream_t>(stream);
  CHECK(grid.x * block.x <= num_random_states) << "current num random state is " << num_random_states << ", less than required " << grid.x * block.x;
  fill_uniform<T><<<grid, block, 0, cu_stream>>>(array, array_len, min, max, random_states,  num_random_states);
}

template<typename T>
void ArrayGenerator::byRepeat(T* array, const T* src_array, size_t src_len, const size_t repeats, StreamHandle stream) {
  if (repeats == 0) return;
  SAM_1D_GRID_INIT(src_len * repeats);
  auto cu_stream = static_cast<cudaStream_t>(stream);
  fill_repeat<><<<grid, block, 0, cu_stream>>>(array, src_array, src_len, repeats);
}

template void ArrayGenerator::byRepeat<IdType>(IdType*, const IdType*, size_t, const size_t, StreamHandle);
template void ArrayGenerator::byUniform<IdType>(IdType*, size_t, IdType, IdType, curandState *, size_t, StreamHandle);


}
}
}