#include "logging.h"
#include "extractor.h"

namespace samgraph {
namespace common {

Extractor::Extractor() {
    if (getenv("SAMGRAPH_OMP_THREAD")) {
        _num_threads = atoi(getenv("SAMGRAPH_OMP_THREAD"));
    } else {
        _num_threads = 40;
    }
}

int Extractor::extract(void *dst, const void *src, const IdType *idx,
                           size_t num_idx, size_t dim, DataType dtype) {
    switch(dtype) {
        case kSamF32:
            return _extract(reinterpret_cast<float *>(dst),
                            reinterpret_cast<const float *>(src),
                            idx, num_idx, dim);
        case kSamI32:
            return _extract(reinterpret_cast<int32_t *>(dst),
                            reinterpret_cast<const int32_t *>(src),
                            idx, num_idx, dim);
        case kSamI64:
            return _extract(reinterpret_cast<int64_t *>(dst),
                            reinterpret_cast<const int64_t *>(src),
                            idx, num_idx, dim);
        default:
            SAM_CHECK(0);
    }

    return 0;
}

template<typename T>
int Extractor::_extract(T* dst, const T* src, const IdType *idx,
                            size_t num_idx, size_t dim) {
    #pragma omp parallel for num_threads(_num_threads)
    for (size_t i = 0; i < num_idx; ++i) {
        #pragma omp simd
        for (size_t j = 0; j < dim; j++) {
            dst[i * dim + j] = src[idx[i] * dim + j];
        }
    }

    return 0;
}

} // namespace common
} // namespace samgraph