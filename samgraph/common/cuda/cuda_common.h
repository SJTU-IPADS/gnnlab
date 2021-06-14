#ifndef SAMGRAPH_CUDA_COMMON_H
#define SAMGRAPH_CUDA_COMMON_H

namespace samgraph {
namespace common {
namespace cuda {

enum QueueType { kGPUSample = 0, kDataCopy, kNumQueues };

const int QueueNum = (int)kNumQueues;

}  // namespace cuda
}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_CUDA_COMMON_H