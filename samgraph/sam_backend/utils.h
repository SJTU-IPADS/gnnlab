#pragma once
#include "model.h"
// #include <cub/cub.cuh>
#include <cusparse.h>

namespace samgraph {
namespace sam_backend {
void sort_coo(TensorPtr row, TensorPtr col, int m, int n, cusparseHandle_t sparse,
              cudaStream_t stream);
bool check_nan_exist(TensorPtr ptr);
} // namespace sam_backend
} // namespace samgraph