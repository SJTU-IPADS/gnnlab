#include "vectorAdd.h"

__global__ void vectorAdd(const float *A, const float *B, float *C,
                          int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

void vectorAddWrapper(int blocksPerGrid, int threadsPerBlock,
                     const float *A, const float *B, float *C,
                     int numElements) {
    vectorAdd <<<blocksPerGrid, threadsPerBlock>>>(A, B, C, numElements);
}
