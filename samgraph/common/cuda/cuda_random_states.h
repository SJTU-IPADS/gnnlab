#ifndef SAMGRAPH_RANDOM_STATES_H
#define SAMGRAPH_RANDOM_STATES_H

#include <curand_kernel.h>

#include <vector>

#include "../common.h"
#include "../constant.h"

namespace samgraph {
namespace common {
namespace cuda {

class GPURandomStates {
 public:
  GPURandomStates(SampleType sample_type, const std::vector<size_t>& fanout,
                  const size_t batch_size, Context ctx);
  ~GPURandomStates();

  curandState* GetStates() { return _states; };
  size_t NumStates() { return _num_states; };

 private:
  curandState* _states;
  size_t _num_states;
  Context _ctx;
};

}  // namespace cuda
}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_RANDOM_STATES_H
