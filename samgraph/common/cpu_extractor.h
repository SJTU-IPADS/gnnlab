#ifndef SAMGRAPH_CPU_EXTRACTOR_H
#define SAMGRAPH_CPU_EXTRACTOR_H

#include "common.h"
#include "types.h"

namespace samgraph {
namespace common {

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

} // namespace common
} // namespace samgraph

#endif // SAMGRAPH_CPU_EXTRACTOR_H