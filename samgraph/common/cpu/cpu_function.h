#ifndef SAMGRAPH_CPU_FUNCTION_H
#define SAMGRAPH_CPU_FUNCTION_H

#include "../common.h"

namespace samgraph {
namespace common {
namespace cpu {

void CPUSample(const IdType *const indptr, const IdType *const indices,
               const IdType *const input, const size_t num_input,
               IdType *output_src, IdType *output_dst, size_t *num_ouput,
               const size_t fanout);

void CPUExtract(void *dst, const void *src, const IdType *index,
                size_t num_index, size_t dim, DataType dtype);

}  // namespace cpu
}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_CPU_FUNCTION_H