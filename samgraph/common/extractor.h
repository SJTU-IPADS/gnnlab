#ifndef SAMGRAPH_EXTRACTOR_H
#define SAMGRAPH_EXTRACTOR_H

#include "common.h"

namespace samgraph {
namespace common {

class Extractor {
 public:
  Extractor();

  int extract(void *dst, const void *src, const IdType *idx, size_t num_idx,
              size_t dim, DataType dtype);

 private:
  template <typename T>
  int _extract(T *dst, const T *src, const IdType *idx, size_t num_idx,
               size_t dim);

  int _num_threads;
};

}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_EXTRACTOR_H