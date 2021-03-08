#ifndef SAMGRAPH_CUDA_COO2CSR_H
#define SAMGRAPH_CUDA_COO2CSR_H

#include <cuda_runtime.h>

#include "types.h"

namespace samgraph {
namespace common {
namespace cuda {

void ConvertCoo2Csr(nodeid_t *src, nodeid_t *dst, int m, int n, int nnz, nodeid_t *indptr, int device, cudaStream_t stream);

} // namespace cuda
} // namespace common
} // namespace samgraph

#endif // SAMGRAPH_CUDA_COO2CSR_H
