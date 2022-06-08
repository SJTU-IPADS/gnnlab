#include "common/cuda/cuda_hashtable.h"

namespace {
using namespace samgraph;
using namespace samgraph::common;
using namespace samgraph::common::cuda;

template <size_t BLOCK_SIZE=Constant::kCudaBlockSize,
          size_t TILE_SIZE=Constant::kCudaTileSize>
__global__ void validate_search(
    const IdType *const unique_data, const size_t num_unique,
    const DeviceOrderedHashTable &table) {
  using Bucket = typename OrderedHashTable::BucketO2N;
  assert(BLOCK_SIZE == blockDim.x);
  const size_t start = threadIdx.x + TILE_SIZE * blockIdx.x;
  const size_t end = min(TILE_SIZE * (blockIdx.x + 1), num_unique);

  for (size_t index = start; index < end; index += BLOCK_SIZE) {
    IdType item = unique_data[index];
    const Bucket &bucket = *table.SearchO2N(item);
    assert(bucket.local == index);
  }
}

}

void ValidateSearch(OrderedHashTable &table, StreamHandle stream) {
  const IdType *unique_data; 
  IdType num_unique;
  table.RefUnique(unique_data, &num_unique);
  const size_t num_tiles = RoundUpDiv((size_t)num_unique, Constant::kCudaTileSize);
  const dim3 grid(num_tiles);
  const dim3 block(Constant::kCudaBlockSize);
  auto cu_stream = static_cast<cudaStream_t>(stream);
  validate_search<><<<grid, block, 0, cu_stream>>>(unique_data, num_unique, table.DeviceHandle());
}
