#pragma once

namespace samgraph {
namespace sam_backend {

constexpr int MAX_NUM_INPUTS = 2;
// constexpr int MAX_NUM_OUTPUTS = 1;

const int CUDA_NUM_THREADS = 512;
const int BLOCK_SIZE_LIMIT = 32768;
inline int GET_BLOCKS(const int N) {
  int ret = (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
  return (ret > BLOCK_SIZE_LIMIT) ? BLOCK_SIZE_LIMIT : ret;
}

enum ModelMode {
  KModeTrain,
  KModeInfer,
};

enum VertexType {
  kVertexTrain,
  kVertexVal,
  kVertexTest,
  kVertexNone,
};

enum AggrType {
  KAggrAvg,
  KAggrMax,
  KAggrMin,
  KAggrSum,
};

enum ActiMode {
  KAcModeNone,
  KAcModeRelu,
  KAcModeSigmoid,
};

enum ElementType {
  kEwTypeAdd,
  kEwTypeMul,
};

enum NormMode {
  kNormModeDirect,
  kNormModeSqrt,
};

} // namespace sam_backend
} // namespace samgraph