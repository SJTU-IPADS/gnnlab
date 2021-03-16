#include <cassert>
#include <cstdio>

#include "cuda_util.h"
#include "config.h"
#include "logging.h"

namespace samgraph {
namespace common {
namespace cuda {

template<typename T, size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void fill(T *data, size_t len, T val) {
    assert(blockDim.x == BLOCK_SIZE);

    const size_t block_start = TILE_SIZE * blockIdx.x;
    const size_t block_end = TILE_SIZE * (blockIdx.x + 1);
    
    for (size_t index = threadIdx.x + block_start; index < block_end; index += BLOCK_SIZE) {
        data[index] = val;
    }
}

template<size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void print_id(const IdType *input, const size_t num_input) {
    assert(blockDim.x == BLOCK_SIZE);

    const size_t block_start = TILE_SIZE * blockIdx.x;
    const size_t block_end = TILE_SIZE * (blockIdx.x + 1);
    
    for (size_t index = threadIdx.x + block_start; index < block_end; index += BLOCK_SIZE) {
        if (index < num_input) {
            printf("PrintID: index %lu is %u\n", index, input[index]);
        }
    }
}

void Fill(float *data, size_t len, float val, cudaStream_t stream) {
    const uint32_t num_tiles = (len + Config::kCudaTileSize - 1) / Config::kCudaTileSize;

    const dim3 grid(num_tiles);
    const dim3 block(Config::kCudaBlockSize);

    fill<float, Config::kCudaBlockSize, Config::kCudaTileSize>
        <<<grid, block, 0, stream>>>(data, len, val);
    CUDA_CALL(cudaStreamSynchronize(stream));
    CUDA_CALL(cudaGetLastError());
}

void PrintID(const IdType *input, const size_t num_input, cudaStream_t stream) {
    const uint32_t num_tiles = (num_input + Config::kCudaTileSize - 1) / Config::kCudaTileSize;

    const dim3 grid(num_tiles);
    const dim3 block(Config::kCudaBlockSize);

    SAM_LOG(DEBUG) << "PrintID: input " << input << " with num_input " << num_input;

    print_id<Config::kCudaBlockSize, Config::kCudaTileSize>
        <<<grid, block, 0, stream>>>(input, num_input);
    CUDA_CALL(cudaStreamSynchronize(stream));
    CUDA_CALL(cudaGetLastError());
}

} // namespace cuda
} // namespace common
} // namespace samgraph