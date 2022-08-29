#pragma once
#include "../run_config.h"
#include <type_traits>

namespace samgraph {
namespace common {
namespace cpu {

template <typename T>
void ArrangeArray(T* array, size_t array_len, T begin = 0, T step = 1) {
  if (array_len == 0) return;
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for(size_t i = 0; i < array_len; i++) {
    array[i] = begin + i * step;
  }
}

// template void ArrangeArray<int>(int*, size_t, int, int);
// template void ArrangeArray<uint64_t>(uint64_t*, size_t, uint64_t, uint64_t);
// template void ArrangeArray<float>(float*, size_t, float, float);

template<typename T>
int CountBits(T a) {
  static_assert(std::is_integral<T>::value, "Integral required.");
  int count = 0;
  while (a) {
    a &= a-1;
    count ++;
  }
  return count;
};

}
}
}