#include <cstdio>
#include <chrono>

#include <cuda_runtime.h>
#include <cusparse.h>
#include <curand.h>
#include <curand_kernel.h>

#include "common.h"

__global__ void rand(unsigned long seed) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    curandState state;
    curand_init(seed, i, 0, &state);

    size_t k = curand(&state) % 100;
    printf("Block %d, thread %d rand %lu\n", blockIdx.x, threadIdx.x, k);
}

int main() {
    dim3 grid(10);
    dim3 block(10);

    unsigned long seed = std::chrono::system_clock::now().time_since_epoch().count();

    CUDA_CALL(cudaSetDevice(0));
    rand<<<grid, block>>>(seed);
    CUDA_CALL(cudaDeviceReset());

    return 0;
}