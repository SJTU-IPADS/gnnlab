#include "common.h"

#include <cuda_runtime.h>

int main() {
    cudaStream_t stream0, stream1;
    void *data0, *data1;
    size_t sz = 10;

    CUDA_CALL(cudaSetDevice(0));
    CUDA_CALL(cudaStreamCreate(&stream0));
    CUDA_CALL(cudaMalloc(&data0, sz));
    
    CUDA_CALL(cudaSetDevice(1));
    CUDA_CALL(cudaStreamCreate(&stream1));
    CUDA_CALL(cudaMalloc(&data1, sz));

    // If the data can be copied by the target stream
    CUDA_CALL(cudaMemcpyAsync(data1, data0, sz, cudaMemcpyDeviceToDevice, stream1));
    CUDA_CALL(cudaStreamSynchronize(stream1));

    // If the stream can be used when the device is set to another device.
    CUDA_CALL(cudaSetDevice(0));
    CUDA_CALL(cudaMemcpyAsync(data1, data0, sz, cudaMemcpyDeviceToDevice, stream1));
    CUDA_CALL(cudaStreamSynchronize(stream1));

    return 0;
}