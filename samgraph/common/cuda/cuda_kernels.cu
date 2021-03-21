#include <cassert>
#include <cstdio>

#include "../config.h"
#include "../logging.h"
#include "cuda_function.h"

namespace samgraph {
namespace common {
namespace cuda {

template<typename T, size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void fill(T *data, size_t len, T val) {
    assert(blockDim.x == BLOCK_SIZE);

    const size_t block_start = TILE_SIZE * blockIdx.x;
    const size_t block_end = TILE_SIZE * (blockIdx.x + 1);
    
    for (size_t index = threadIdx.x + block_start; index < block_end; index += BLOCK_SIZE) {
        if (index < len) {
            data[index] = val;
        }
    }
}

void Fill(float *data, size_t len, float val, cudaStream_t stream, bool sync) {
    const uint32_t num_tiles = (len + Config::kCudaTileSize - 1) / Config::kCudaTileSize;

    const dim3 grid(num_tiles);
    const dim3 block(Config::kCudaBlockSize);

    fill<float, Config::kCudaBlockSize, Config::kCudaTileSize>
        <<<grid, block, 0, stream>>>(data, len, val);
    
    if (sync) {
        CUDA_CALL(cudaStreamSynchronize(stream));
    }
}

} // namespace cuda
} // namespace common
} // namespace samgraph