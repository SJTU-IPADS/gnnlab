#include <cassert>
#include <chrono>
#include "../common.h"
#include "../device.h"
#include "../logging.h"
#include "../timer.h"
#include "cuda_random_seeder.h"
namespace samgraph {
namespace common {
namespace cuda {

static __global__ void seeds_init(curandState *states, size_t num, unsigned long seed) {
  size_t threadId = threadIdx.x + blockIdx.x * blockDim.x;
  if (threadId >= num) {
    return;
  }
  curand_init(seed, threadId, 0, &states[threadId]);
}

void GPURandomSeeder::Init(std::vector<int> fanouts, Context sampler_ctx, 
                           StreamHandle sampler_stream, size_t batch_size) {
  if (_initialize) {
      return;
  }
  Timer t1;
  cudaStream_t cu_stream = static_cast<cudaStream_t>(sampler_stream);
  auto sampler_device = Device::Get(sampler_ctx);
  // get maximum number of curandState usage
  long num_random_t = batch_size;
  for (auto i : fanouts) {
      num_random_t *= i;
  }
  if (num_random_t >= 0xffffffff) {
    LOG(FATAL) << "Sampling random seed size is too large";
  }
  _num_random = static_cast<size_t>(num_random_t);
  if (_num_random > maxSeedNum) {
    _num_random = maxSeedNum;
  }

  _states = static_cast<curandState*>(
          sampler_device->AllocDataSpace(sampler_ctx, sizeof(curandState) * _num_random));
  const size_t blockSize = Constant::kCudaBlockSize;
  const dim3 grid((_num_random + blockSize - 1) / blockSize);
  const dim3 block(Constant::kCudaBlockSize);
  unsigned long seed = std::chrono::system_clock::now().time_since_epoch().count();
  seeds_init <<<grid, block, 0, cu_stream>>>(_states, _num_random, seed);
  sampler_device->StreamSync(sampler_ctx, sampler_stream);

  _initialize = true;
  double random_seeder_init_time = t1.Passed();
  LOG(DEBUG) << "GPURandomSeeder initialized " << _num_random << " seeds, random initialization coast time: " 
             << random_seeder_init_time;
}

} // cuda
} // common
} // samgraph
