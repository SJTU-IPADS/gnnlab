#include "cpu_extractor.h"
#include "logging.h"

namespace samgraph {
namespace common {

CpuExtractor::CpuExtractor() {
    if (getenv("SAMGRAPH_OMP_THREAD")) {
        _num_threads = atoi(getenv("SAMGRAPH_OMP_THREAD"));
    } else {
        _num_threads = 4;
    }
}

int CpuExtractor::extract(void *dst, const void *src, const nodeid_t *idx,
                           size_t num_idx, size_t dim, DataType dtype) {
    switch(dtype) {
        case kSamF32:
            return _extract(reinterpret_cast<float *>(dst),
                            reinterpret_cast<const float *>(src),
                            idx, num_idx, dim);
        default:
            SAM_CHECK(0);
    }

    return 0;
}

template<typename T>
int CpuExtractor::_extract(T* dst, const T* src, const nodeid_t *idx,
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