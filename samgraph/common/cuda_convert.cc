#include <cusparse.h>

#include "cuda_convert.h"
#include "logging.h"

namespace samgraph {
namespace common {
namespace cuda {

void ConvertCoo2Csr(nodeid_t *src, nodeid_t *dst, int m, int n, int nnz, nodeid_t *indptr, int device, cudaStream_t stream) {
    CUDA_CALL(cudaSetDevice(device));

    cusparseHandle_t handle;
    CUSPARSE_CALL(cusparseCreate(&handle));
    CUSPARSE_CALL(cusparseSetStream(handle, stream));

    // Sort coo by row in place
    size_t p_buffer_size;
    void *p_buffer;
    int *P;

    CUSPARSE_CALL(cusparseXcoosort_bufferSizeExt(handle, m, n, nnz, src, dst, &p_buffer_size));
    CUDA_CALL(cudaMalloc(&p_buffer, sizeof(char) * p_buffer_size));

    CUDA_CALL(cudaMalloc((void **)&P, sizeof(int) * nnz));
    CUSPARSE_CALL(cusparseCreateIdentityPermutation(handle, nnz, P));

    CUSPARSE_CALL(cusparseXcoosortByRow(handle, m, n , nnz, src, dst, P, p_buffer));

    CUDA_CALL(cudaFree(p_buffer));
    CUDA_CALL(cudaFree(P));

    // Convert coo 2 csr
    CUSPARSE_CALL(cusparseXcoo2csr(handle, src, nnz, m, indptr, CUSPARSE_INDEX_BASE_ZERO));
    
    CUSPARSE_CALL(cusparseDestroy(handle));
}

} // namespace cuda
} // namespace common
} // namespace samgraph
