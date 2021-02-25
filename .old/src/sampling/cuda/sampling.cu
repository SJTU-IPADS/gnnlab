#include <curand.h>
#include <curand_kernel.h>

#include "sampling/cuda/wrapper.hpp"

__global__ void sampleBlock(const unsigned int *indptr,
                            const unsigned int *indices,
                            const unsigned int *input,
                            unsigned int *output,
                            unsigned int num_input,
                            unsigned int fanout,
                            unsigned int seed) {
    const size_t block_start = 1024 * blockIdx.x;
    const size_t block_end = 1024 * (blockIdx.x + 1);

    int seq = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(seed, seq, 0, &state);

    if (threadIdx.y < fanout) {
        for (size_t index = threadIdx.x + block_start; index < block_end; index += 256) {
            if (threadIdx.x < num_input) {
                const unsigned int start = indptr[index];
                const unsigned int end = indptr[index + 1];

                const unsigned int num_nbr = end - start;
                for (int n = 0; n < fanout; n++) {
                    if (num_nbr == 0) {
                        output[index * fanout + n] = 0xffffffff;
                    } else {
                        output[index * fanout + n] = curand(&state) % num_nbr;
                    }
                }
            }
        }
    }
}

void sampleBlockWrapper(const unsigned int *indptr, const unsigned int *indices,
                        const unsigned int *input,
                        unsigned int *output,
                        unsigned int num_input,
                        unsigned int fanout,
                        unsigned int seed) {
    const size_t num_tiles = (num_input + 1024 - 1) / 1024;
    
    const dim3 grid(num_tiles);
    const dim3 block(256);

    sampleBlock<<<grid, block>>>(
        indptr,
        indices,
        input,
        output,
        num_input,
        fanout,
        seed
    );
}
