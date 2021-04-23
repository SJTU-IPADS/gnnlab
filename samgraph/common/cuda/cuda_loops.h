#ifndef SAMGRAPH_CUDA_LOOPS_H
#define SAMGRAPH_CUDA_LOOPS_H

namespace samgraph {
namespace common {
namespace cuda {

void GPUSampleLoop();
void DataCopyLoop();

bool RunGPUSampleLoopOnce();
bool RunDataCopyLoopOnce();

}  // namespace cuda
}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_CUDA_LOOPS_H