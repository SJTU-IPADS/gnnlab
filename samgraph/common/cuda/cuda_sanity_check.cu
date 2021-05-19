#include "../constant.h"
#include "../device.h"
#include "cuda_function.h"

namespace samgraph {
namespace common {
namespace cuda {

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void batch_sanity_check(IdType *map, const IdType *input,
                                   const size_t num_input) {
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_input) {
      if (map[input[index]] > 0) {
        printf("duplicate batch input");
      }
      assert(map[input[index]] == 0);
      map[input[index]] = 1;
    }
  }
}

void GPUBatchSanityCheck(IdType *map, const IdType *input,
                         const size_t num_input, Context ctx,
                         StreamHandle stream) {
  auto device = Device::Get(ctx);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  const uint32_t num_tiles =
      (num_input + Constant::kCudaTileSize - 1) / Constant::kCudaTileSize;

  const dim3 grid(num_tiles);
  const dim3 block(Constant::kCudaBlockSize);

  batch_sanity_check<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(map, input, num_input);

  device->StreamSync(ctx, stream);
}

} // namespace cuda
} // namespace common
} // namespace samgraph
