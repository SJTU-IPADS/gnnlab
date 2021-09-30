#pragma once

#include "../common.h"
#include "../constant.h"
#include "../cuda/cuda_shuffler.h"
namespace samgraph {
namespace common {
namespace dist {

class PreSampler {
 public:
  PreSampler(TensorPtr input, size_t batch_size, size_t num_nodes);
  ~PreSampler();
  void DoPreSample();
  TensorPtr GetFreq();
  TensorPtr GetRankNode();
  void GetRankNode(TensorPtr &);
  static inline void SetSingleton(PreSampler* p) { singleton = p; }
  static inline PreSampler* Get() { return singleton; }
  TaskPtr DoPreSampleShuffle();
 private:
  Id64Type * freq_table;
  // TensorPtr freq_table;
  size_t _num_nodes, _num_step;
  static PreSampler* singleton;
  cuda::GPUShuffler* _shuffler;

};

}
}
}
