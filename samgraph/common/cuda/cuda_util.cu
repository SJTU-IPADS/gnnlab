#include <cassert>
#include <cstdio>

#include "../config.h"
#include "../logging.h"
#include "cuda_util.h"

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

template<typename T, size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void assert_val(const T *input, const size_t num_input, const T min, const T max) {
    assert(blockDim.x == BLOCK_SIZE);

    const size_t block_start = TILE_SIZE * blockIdx.x;
    const size_t block_end = TILE_SIZE * (blockIdx.x + 1);
    
    for (size_t index = threadIdx.x + block_start; index < block_end; index += BLOCK_SIZE) {
        if (index < num_input) {
            assert(input[index] >= min && input[index] < max);
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

void PrintID(const IdType *input, const size_t num_input, cudaStream_t stream) {
    const uint32_t num_tiles = (num_input + Config::kCudaTileSize - 1) / Config::kCudaTileSize;

    const dim3 grid(num_tiles);
    const dim3 block(Config::kCudaBlockSize);

    SAM_LOG(DEBUG) << "PrintID: input " << input << " with num_input " << num_input;

    print_id<Config::kCudaBlockSize, Config::kCudaTileSize>
        <<<grid, block, 0, stream>>>(input, num_input);
    CUDA_CALL(cudaStreamSynchronize(stream));
}

void AssertVal(const int64_t *input, const size_t num_input, const int64_t min, const int64_t max, cudaStream_t stream) {
    const uint32_t num_tiles = (num_input + Config::kCudaTileSize - 1) / Config::kCudaTileSize;

    const dim3 grid(num_tiles);
    const dim3 block(Config::kCudaBlockSize);

    assert_val<int64_t, Config::kCudaBlockSize, Config::kCudaTileSize>
        <<<grid, block, 0, stream>>>(input, num_input, min, max);
    CUDA_CALL(cudaStreamSynchronize(stream));
}

} // namespace cuda
} // namespace common
} // namespace samgraph