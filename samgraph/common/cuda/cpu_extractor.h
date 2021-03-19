#ifndef SAMGRAPH_CUDA_CPU_EXTRACTOR_H
#define SAMGRAPH_CUDA_CPU_EXTRACTOR_H

#include "../common.h"
#include "../types.h"

namespace samgraph {
namespace common {
namespace cuda {

class CpuExtractor {
 public:
  CpuExtractor();

  int extract(void *dst, const void *src, const IdType *idx,
               size_t num_idx, size_t dim, DataType dtype);
 private:
  template <typename T>
  int _extract(T* dst, const T* src, const IdType *idx,
               size_t num_idx, size_t dim);

  int _num_threads;
};

} // namesapce cuda
} // namespace common
} // namespace samgraph

#endif // SAMGRAPH_CUDA_CPU_EXTRACTOR_H