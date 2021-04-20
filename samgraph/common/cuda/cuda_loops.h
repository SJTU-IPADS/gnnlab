#ifndef SAMGRAPH_CUDA_LOOPS_H
#define SAMGRAPH_CUDA_LOOPS_H

namespace samgraph {
namespace common {
namespace cuda {

void GpuSampleLoop();
void DataCopyLoop();

bool RunGpuSampleLoopOnce();
bool RunDataCopyLoopOnce();

}  // namespace cuda
}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_CUDA_LOOPS_H