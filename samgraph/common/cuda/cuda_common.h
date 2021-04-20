#ifndef SAMGRAPH_CUDA_COMMON_H
#define SAMGRAPH_CUDA_COMMON_H

namespace samgraph {
namespace common {
namespace cuda {

enum QueueType { kGpuSample, kDataCopy, kLastFakeQueue };

const int QueueNum = (int)kLastFakeQueue;

}  // namespace cuda
}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_CUDA_COMMON_H