#ifndef SAMGRAPH_RANDOM_STATES_H
#define SAMGRAPH_RANDOM_STATES_H

#include <curand_kernel.h>

#include <vector>

namespace samgraph {
namespace common {
namespace cuda {

class GPURandomStates {
 public:
  GPURandomStates(std::vector<int> fanouts, size_t batch_size,
                  Context sampler_ctx);

  curandState* Get() { return _states; };
  size_t Size() { return _num_random; };

  static constexpr size_t maxSeedNum =
      ((5l * 1024 * 1024 + Constant::kCudaBlockSize - 1) /
       Constant::kCudaBlockSize * Constant::kCudaBlockSize);

 private:
  // random seeds list in CUDA for sampling
  curandState* _states;
  // random seeds size
  size_t _num_random;
};

}  // namespace cuda
}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_RANDOM_STATES_H
