#ifndef SAMGRAPH_CUDA_FUNCTION_H
#define SAMGRAPH_CUDA_FUNCTION_H

#include <cuda_runtime.h>
#include <cusparse.h>

#include "../common.h"
#include "cuda_hashtable.h"

namespace samgraph{
namespace common {
namespace cuda {

void CudaSample(const IdType *indptr, const IdType *indices, const IdType *input, const size_t num_input,
                  const size_t fanout, IdType *out_src, IdType *out_dst, size_t *num_out, cudaStream_t stream);

void MapEdges(const IdType * const global_src,
              IdType * const new_global_src,
              const IdType * const global_dst,
              IdType * const new_global_dst,
              const size_t num_edges,
              DeviceOrderedHashTable mapping,
              cudaStream_t stream);

void ConvertCoo2Csr(IdType *src, IdType *dst, int m, int n, int nnz, IdType *indptr,
                    int device, cusparseHandle_t handle, cudaStream_t stream);

void Fill(float *data, size_t len, float val, cudaStream_t stream, bool sync=false);

} // namespace cuda
} // namespace common
} // namespace samgraph


#endif // SAMGRAPH_CUDA_FUNCTION_H