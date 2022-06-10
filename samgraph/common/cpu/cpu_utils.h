#pragma once
#include "../run_config.h"

namespace samgraph {
namespace common {
namespace cpu {

template <typename T>
void ArrangeArray(T* array, size_t array_len, T begin = 0, T step = 1) {
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for(size_t i = 0; i < array_len; i++) {
    array[i] = begin + i * step;
  }
}

template void ArrangeArray<int>(int*, size_t, int, int);

}
}
}