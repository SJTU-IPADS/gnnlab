#ifndef SAMGRAPH_CUDA_PROFILER_H
#define SAMGRAPH_CUDA_PROFILER_H

#include <vector>

namespace samgraph {
namespace common {
namespace cuda {

class GPUProfiler {
 public:
  std::vector<size_t> sample_time;
  std::vector<size_t> copy_time;

 private:
  size_t _num_entries;
};

}  // namespace cuda
}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_CUDA_RPOFILER_H