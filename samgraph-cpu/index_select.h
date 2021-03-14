#pragma once

#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cstring>

template<typename DTYPE>
DTYPE* IndexSelect(const DTYPE *feature, size_t dim, const std::vector<uint32_t> &index) {
    DTYPE *ret = (DTYPE *) malloc(index.size() * dim * sizeof(DTYPE));
#pragma omp parallel for
    for (uint32_t i = 0; i < index.size(); i++) {
        memcpy(ret + i * dim, feature + index[i] * dim, dim * sizeof(DTYPE));
    }

    return ret;
}
