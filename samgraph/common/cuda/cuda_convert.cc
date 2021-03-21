#include <cusparse.h>

#include "../logging.h"
#include "cuda_function.h"

namespace samgraph {
namespace common {
namespace cuda {

void ConvertCoo2Csr(IdType *src, IdType *dst, int m, int n, int nnz, IdType *indptr,
                    int device, cusparseHandle_t handle, cudaStream_t stream) {
    SignedIdType *signed_src = static_cast<SignedIdType*>(static_cast<void *>(src));
    SignedIdType *signed_dst = static_cast<SignedIdType*>(static_cast<void *>(dst));
    SignedIdType *signed_indptr = static_cast<SignedIdType*>(static_cast<void*>(indptr));

    // Sort coo by row in place
    size_t p_buffer_size;
    void *p_buffer;
    int *P;

    CUSPARSE_CALL(cusparseXcoosort_bufferSizeExt(handle, m, n, nnz, signed_src, signed_dst, &p_buffer_size));
    CUDA_CALL(cudaMalloc(&p_buffer, sizeof(char) * p_buffer_size));

    CUDA_CALL(cudaMalloc((void **)&P, sizeof(int) * nnz));
    CUSPARSE_CALL(cusparseCreateIdentityPermutation(handle, nnz, P));

    CUSPARSE_CALL(cusparseXcoosortByRow(handle, m, n , nnz, signed_src, signed_dst, P, p_buffer));

    // Convert coo 2 csr
    CUSPARSE_CALL(cusparseXcoo2csr(handle, signed_src, nnz, m, signed_indptr, CUSPARSE_INDEX_BASE_ZERO));

    CUDA_CALL(cudaStreamSynchronize(stream));
    CUDA_CALL(cudaFree(p_buffer));
    CUDA_CALL(cudaFree(P));
}

} // namespace cuda
} // namespace common
} // namespace samgraph
