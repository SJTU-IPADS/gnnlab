#pragma once
#include "cpu/cpu_utils.h"
#include "cuda/cuda_utils.h"

namespace samgraph {
namespace common {

template <typename T>
void ArrangeArray(Context ctx, T* array, size_t array_len, T begin = 0, T step = 1, StreamHandle stream = nullptr) {
  switch (ctx.device_type) {
    case kMMAP:
    case kCPU: cpu::ArrangeArray(array, array_len, begin, step); break;
    case kGPU: cuda::ArrangeArray(array, array_len, begin, step, stream); break;
    default:
      CHECK(false);
  }
}
}
}