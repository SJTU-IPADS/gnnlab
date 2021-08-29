#pragma once

#include "../common.h"
#include "../constant.h"
namespace samgraph {
namespace common {
namespace cuda {

class PreSampler {
 public:
  PreSampler(size_t num_nodes, size_t num_step);
  TensorPtr DoPreSample();
  TensorPtr GetFreq();
  TensorPtr GetRankNode();
  static inline void SetSingleton(PreSampler* p) { singleton = p; }
  static inline PreSampler* Get() { return singleton; }
 private:
  Id64Type * freq_table;
  // TensorPtr freq_table;
  size_t _num_nodes, _num_step;
  static PreSampler* singleton;

};

}
}
}