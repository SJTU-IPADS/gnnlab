#ifndef SAMGRAPH_EXTRACTOR_H
#define SAMGRAPH_EXTRACTOR_H

#include "common.h"

namespace samgraph {
namespace common {

class Extractor {
 public:
  void Extract(void *dst, const void *src, const IdType *idx, size_t num_idx,
               size_t dim, DataType dtype);

 private:
  template <typename T>
  void DoExtract(T *dst, const T *src, const IdType *idx, size_t num_idx,
                 size_t dim);
};

}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_EXTRACTOR_H