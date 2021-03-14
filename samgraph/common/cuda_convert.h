#ifndef SAMGRAPH_CUDA_COO2CSR_H
#define SAMGRAPH_CUDA_COO2CSR_H

#include <cuda_runtime.h>
#include <cusparse.h>

#include "types.h"

namespace samgraph {
namespace common {
namespace cuda {

void ConvertCoo2Csr(IdType *src, IdType *dst, int m, int n, int nnz, IdType *indptr,
                    int device, cusparseHandle_t handle, cudaStream_t stream);

} // namespace cuda
} // namespace common
} // namespace samgraph

#endif // SAMGRAPH_CUDA_COO2CSR_H
