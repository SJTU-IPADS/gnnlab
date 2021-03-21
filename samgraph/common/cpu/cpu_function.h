#ifndef SAMGRAPH_CPU_FUNCTION_H
#define SAMGRAPH_CPU_FUNCTION_H

#include "../common.h"

namespace samgraph {
namespace common {
namespace cpu {

void CpuSample(const IdType *const indptr, const IdType *const indices,
               const IdType *const input, const size_t num_input,
               IdType *output_src, IdType *output_dst,
               size_t *num_ouput, const size_t fanout);

void ConvertCoo2Csr(const IdType *src, const IdType *dst,
                    IdType *out_indptr, IdType *out_indices,
                    const size_t m, const size_t nnz);

} // namespace cpu
} // namespace common
} // namespace samgraph


#endif // SAMGRAPH_CPU_FUNCTION_H