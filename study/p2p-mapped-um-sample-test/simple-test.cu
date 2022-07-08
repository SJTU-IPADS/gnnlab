#include <bits/stdc++.h>
#include "utility.h"
using namespace std;

int main() {
    cudaStream_t stream;
    cudaEvent_t start, end;
    volatile int* flag;

    int *arr, *result;
    int len = 1;

    CUDA_CALL(cudaSetDevice(0));
    CUDA_CALL(cudaMalloc(&arr, sizeof(int) * len));
    CUDA_CALL(cudaMalloc(&result, sizeof(int) * len));
    CUDA_CALL(cudaMemset(arr, 0, sizeof(int) * len));
    CUDA_CALL(cudaMemset(result, 0, sizeof(int) * len));
    CUDA_CALL(cudaHostAlloc(&flag, sizeof(int), cudaHostAllocPortable));
    CUDA_CALL(cudaStreamCreate(&stream));
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&end));

    int r = 100;

    CUDA_CALL(cudaStreamSynchronize(stream));
    *flag = 0;
    delay<<<1, 1, 0, stream>>>(flag);
    CUDA_CALL(cudaEventRecord(start, stream));
    for(int i = 0; i < r; i++) {
        arr += 0;
        read<<<1, 1, 0, stream>>>(arr, len, result, len);
    }
    CUDA_CALL(cudaEventRecord(end, stream));
    *flag = 1;
    CUDA_CALL(cudaStreamSynchronize(stream));
    
    float ms;
    CUDA_CALL(cudaEventElapsedTime(&ms, start, end));
    cout << ms * 1000 / r << "\n";

    CUDA_CALL(cudaFree(arr));
    CUDA_CALL(cudaFree(result));
    CUDA_CALL(cudaFreeHost((void*)flag));
    CUDA_CALL(cudaStreamDestroy(stream));
    CUDA_CALL(cudaEventDestroy(start));
    CUDA_CALL(cudaEventDestroy(end));
}