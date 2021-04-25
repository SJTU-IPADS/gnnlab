#include "extractor.h"

#include "logging.h"
#include "run_config.h"

namespace samgraph {
namespace common {

void Extractor::Extract(void *dst, const void *src, const IdType *idx,
                        size_t num_idx, size_t dim, DataType dtype) {
  switch (dtype) {
    case kF32:
      DoExtract(reinterpret_cast<float *>(dst),
                reinterpret_cast<const float *>(src), idx, num_idx, dim);
      break;
    case kI32:
      DoExtract(reinterpret_cast<int32_t *>(dst),
                reinterpret_cast<const int32_t *>(src), idx, num_idx, dim);
      break;
    case kI64:
      DoExtract(reinterpret_cast<int64_t *>(dst),
                reinterpret_cast<const int64_t *>(src), idx, num_idx, dim);
      break;
    default:
      CHECK(0);
  }
}

template <typename T>
void Extractor::DoExtract(T *dst, const T *src, const IdType *idx,
                          size_t num_idx, size_t dim) {
#pragma omp parallel for num_threads(RunConfig::kOMPThreadNum)
  for (size_t i = 0; i < num_idx; ++i) {
#pragma omp simd
    for (size_t j = 0; j < dim; j++) {
      dst[i * dim + j] = src[idx[i] * dim + j];
    }
  }
}

}  // namespace common
}  // namespace samgraph