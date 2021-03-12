#include <chrono>
#include <cassert>
#include <iostream>

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <curand.h>
#include <curand_kernel.h>

#include "common.h"

constexpr int kEmptyKey = -1;
constexpr size_t kTileSize = 5;
constexpr size_t kBlockSize = 5;

template<size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void
sample(const int *indptr, const int *indices, const int *input, const size_t num_input,
       const size_t fanout, int *tmp_src, int *tmp_dst, unsigned long seed) {
    assert(BLOCK_SIZE == blockDim.x);
    const size_t block_start = TILE_SIZE * blockIdx.x;
    const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    curandState state;
    curand_init(seed, i, 0, &state);

    #pragma unroll
    for (size_t index = threadIdx.x + block_start; index < block_end; index += BLOCK_SIZE) {
        if (index < num_input) {
            const int rid = input[index];
            const int off = indptr[rid];
            const int len = indptr[rid + 1] - indptr[rid];

            if (len <= fanout) {
                size_t j = 0;
                for (; j < len; ++j) {
                    tmp_src[index * fanout + j] = rid;
                    tmp_dst[index * fanout + j] = indices[off + j];
                }

                for (; j < fanout; ++j) {
                    tmp_src[index * fanout + j] = kEmptyKey;
                    tmp_dst[index * fanout + j] = kEmptyKey;
                }
            } else {
                for (size_t j = 0; j < fanout; ++j) {
                    tmp_src[index * fanout + j] = rid;
                    tmp_dst[index * fanout + j] = indices[off + j];
                }

                for (size_t j = fanout; j < len; ++j) {
                    size_t k = curand(&state) % (j + 1);
                    if (k < fanout) {
                        tmp_dst[index * fanout + k] = indices[off + j];
                    }
                } 
            }
        }
    }
}

int main() {
    // constexpr int device = 0;
    
    constexpr size_t m = 30;
    constexpr size_t nnz = 120;
    constexpr size_t num_input = 30;
    constexpr size_t fanout = 5;

    int indptr[m + 1] = {0,2,10,12,20,30,35,38,42,55,60,61,63,68,75,80,82,85,90,95,98,101,104,104,105,107,110,115,118,119,120};
    int indices[nnz] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,
                        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,
                        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,
                        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29};

    int input[m] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29};
    int tmp_src[num_input * fanout];
    int tmp_dst[num_input * fanout];

    int *d_indptr;
    int *d_indices;
    int *d_input;
    int *d_tmp_src;
    int *d_tmp_dst;

    CUDA_CALL(cudaMalloc(&d_indptr, sizeof(int) * (m + 1)));
    CUDA_CALL(cudaMalloc(&d_indices, sizeof(int) * nnz));
    CUDA_CALL(cudaMalloc(&d_input, sizeof(int) * m));
    CUDA_CALL(cudaMalloc(&d_tmp_src, sizeof(int) * num_input * fanout));
    CUDA_CALL(cudaMalloc(&d_tmp_dst, sizeof(int) * num_input * fanout));

    CUDA_CALL(cudaMemcpy(d_indptr, indptr, sizeof(int) * (m + 1), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_indices, indices, sizeof(int) * nnz, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_input, input, sizeof(int) * num_input, cudaMemcpyHostToDevice));

    const size_t num_tiles = (num_input + kTileSize - 1) / kTileSize;
    const dim3 grid(num_tiles);
    const dim3 block(kBlockSize);

    unsigned long seed = std::chrono::system_clock::now().time_since_epoch().count();

    sample<kBlockSize, kTileSize>
        <<<grid, block>>>(d_indptr, d_indices, d_input, num_input,
                          fanout, d_tmp_src, d_tmp_dst, seed);
    
    CUDA_CALL(cudaMemcpy(tmp_src, d_tmp_src, sizeof(int) * num_input * fanout, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(tmp_dst, d_tmp_dst, sizeof(int) * num_input * fanout, cudaMemcpyDeviceToHost));

    for (int i = 0; i < (num_input * fanout); i++) {
        std::cout << tmp_src[i] << ' ';
    }
    std::cout << std::endl;

    for (int i = 0; i < (num_input * fanout); i++) {
        std::cout << tmp_dst[i] << ' ';
    }
    std::cout << std::endl;

    CUDA_CALL(cudaFree(d_indptr));
    CUDA_CALL(cudaFree(d_indices));
    CUDA_CALL(cudaFree(d_input));
    CUDA_CALL(cudaFree(d_tmp_src));
    CUDA_CALL(cudaFree(d_tmp_dst));
}