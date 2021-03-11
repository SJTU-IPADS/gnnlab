#include <cuda_runtime.h>
#include <cusparse.h>

#include "common.h"

int main() {
    constexpr int device = 0;
    
    constexpr int m = 3;
    constexpr int n = 1;
    constexpr int k = 4;

    constexpr int nnz = 6;
    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;

    int indptr[m + 1] = {0,2,4,6};
    int indices[nnz] = {0,1,2,3,0,1};
    float val[nnz] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float input[m] = {1.0f, 2.0f, 3.0f}; 
    float output[k];

    int *d_indptr;
    int *d_indices;
    float *d_val;
    float *d_input;
    float *d_output;

    int ldb = m;
    int ldc = k;

    CUDA_CALL(cudaSetDevice(device));
    CUDA_CALL(cudaMalloc(&d_indptr, (m+1) * sizeof(int)));
    CUDA_CALL(cudaMalloc(&d_indices, nnz * sizeof(int)));
    CUDA_CALL(cudaMalloc(&d_val, nnz * sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_input, m * sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_output, k * sizeof(float)));
    
    CUDA_CALL(cudaMemcpy(d_indptr, indptr, (m+1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_indices, indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_val, val, nnz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_input, input, m * sizeof(float), cudaMemcpyHostToDevice));

    cusparseHandle_t handle;
    cusparseMatDescr_t descr_a;

    CUSPARSE_CALL(cusparseCreate(&handle));
    CUSPARSE_CALL(cusparseCreateMatDescr(&descr_a));
    CUSPARSE_CALL(cusparseSetMatType(descr_a, CUSPARSE_MATRIX_TYPE_GENERAL));
    CUSPARSE_CALL(cusparseSetMatIndexBase(descr_a, CUSPARSE_INDEX_BASE_ZERO));

    CUSPARSE_CALL(cusparseScsrmm(handle,
                  CUSPARSE_OPERATION_TRANSPOSE,
                  m, n, k, nnz, &alpha,
                  descr_a,
                  d_val,
                  d_indptr,
                  d_indices,
                  d_input,
                  ldb,
                  &beta,
                  d_output,
                  ldc));

    CUDA_CALL(cudaMemcpy(output, d_output, k * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < k; i++) {
        std::cout << output[i] << std::endl;
    }

    CUDA_CALL(cudaFree(d_indptr));
    CUDA_CALL(cudaFree(d_indices));
    CUDA_CALL(cudaFree(d_val));
    CUDA_CALL(cudaFree(d_input));
    CUDA_CALL(cudaFree(d_output));

    CUSPARSE_CALL(cusparseDestroyMatDescr(descr_a));
    CUSPARSE_CALL(cusparseDestroy(handle));

    return 0;
}
