#include <cstdint>

#include "cuda_hashtable.hpp"

int main() {
    unsigned int num_input = 5;
    uint32_t input [] = {0, 2, 4, 100, 101};
    uint32_t *d_input;
    uint32_t *d_output;
    uint32_t *d_num_output;

    CUDA_CALL2(cudaMalloc(&d_input, num_input * sizeof(uint32_t)));
    CUDA_CALL2(cudaMalloc(&d_output, num_input * sizeof(uint32_t)));
    CUDA_CALL2(cudaMalloc(&d_num_output, sizeof(uint32_t)));

    CUDA_CALL2(cudaMemcpy(d_input, input, num_input * sizeof(uint32_t), cudaMemcpyHostToDevice));


    OrderedHashTable table(num_input);
    table.FillWithDuplicates(d_input, num_input, d_output, d_num_output);
}