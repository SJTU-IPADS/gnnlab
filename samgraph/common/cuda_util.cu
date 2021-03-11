#include <cassert>

#include "cuda_util.h"
#include "config.h"

namespace samgraph {
namespace common {
namespace cuda {

template<typename T, int BLOCK_SIZE, int TILE_SIZE>
__global__ void fill(T *data, size_t len, T val) {
    assert(blockDim.x == BLOCK_SIZE);

    const size_t block_start = TILE_SIZE * blockIdx.x;
    const size_t block_end = TILE_SIZE * (blockIdx.x + 1);
    
    for (size_t index = threadIdx.x + block_start; index < block_end; index += BLOCK_SIZE) {
        data[index] = val;
    }
}

template<typename T>
void Fill(T *data, size_t len, T val, cudaStream_t stream) {
    const uint32_t num_tiles = (len + Config::kCudaTileSize - 1) / Config::kCudaTileSize;

    const dim3 grid(num_tiles);
    const dim3 block(Config::kCudaBlockSize);

    fill<T, Config::kCudaBlockSize, Config::kCudaTileSize>
        <<<grid, block, 0, stream>>>(data, len, val);
}

}
}
}