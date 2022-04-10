#ifndef SAMGRAPH_CPU_PRE_SAMPLER
#define SAMGRAPH_CPU_PRE_SAMPLER

#include "../common.h"
#include "../constant.h"

namespace samgraph{
namespace common {
namespace cuda {

// used for cache graph in gpu when UM is enabled
class UMPreSampler {
public:
  UMPreSampler(size_t num_nodes, size_t num_step);
  void DoPreSample();
  TensorPtr GetRankNode() const;
private:
  TensorPtr _freq_table_ts;
  size_t _num_nodes;
  size_t _num_step;
};
  
} // namespace cuda
} // namespace common
} // namespace samgraph

#endif