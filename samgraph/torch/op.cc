#include <torch/torch.h>
#include <torch/extension.h>
#include <cusparse.h>
#include <cuda_runtime.h>

#include "ops.h"

PYBIND11_MODULE(c_lib, m) {
    m.def("samgraph.torch.");
}
