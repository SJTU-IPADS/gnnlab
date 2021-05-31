#include <vector>
#include <curand_kernel.h>
namespace samgraph {
namespace common {
namespace cuda {

class GPURandomSeeder {
 public:
  GPURandomSeeder() { states = nullptr; _initialize = false; };
  void Init(std::vector<int> fanouts, Context sampler_ctx,
                             StreamHandle sampler_stream, size_t batch_size);
  curandState* Get() { return states; };

 private:
  // Whether the seeder is initialized
  bool _initialize;
  // random seeds list in CUDA for sampling
  curandState* states;
};

} // cuda
} // common
} // samgraph
