#include "../common.h"
#include "../logging.h"
#include "../run_config.h"
#include "cpu_function.h"

namespace samgraph {
namespace common {
namespace cpu {

namespace {

template <typename T>
void cpu_extract(void *dst, const void *src, const IdType *index,
                 size_t num_index, size_t dim) {
  T *dst_data = reinterpret_cast<T *>(dst);
  const T *src_data = reinterpret_cast<const T *>(src);

#pragma omp parallel for num_threads(RunConfig::kOMPThreadNum)
  for (size_t i = 0; i < num_index; ++i) {
#pragma omp simd
    for (size_t j = 0; j < dim; j++) {
      dst_data[i * dim + j] = src_data[index[i] * dim + j];
    }
  }
}

}  // namespace

void CPUExtract(void *dst, const void *src, const IdType *index,
                size_t num_index, size_t dim, DataType dtype) {
  switch (dtype) {
    case kF32:
      cpu_extract<float>(dst, src, index, num_index, dim);
      break;
    case kF64:
      cpu_extract<double>(dst, src, index, num_index, dim);
      break;
    case kF16:
      cpu_extract<short>(dst, src, index, num_index, dim);
      break;
    case kU8:
      cpu_extract<uint8_t>(dst, src, index, num_index, dim);
      break;
    case kI32:
      cpu_extract<int32_t>(dst, src, index, num_index, dim);
      break;
    case kI64:
      cpu_extract<int64_t>(dst, src, index, num_index, dim);
      break;
    default:
      CHECK(0);
  }
}

}  // namespace cpu
}  // namespace common
}  // namespace samgraph