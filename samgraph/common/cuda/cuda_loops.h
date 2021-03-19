#ifndef SAMGRAPH_CUDA_LOOPS_H
#define SAMGRAPH_CUDA_LOOPS_H

namespace samgraph {
namespace common {
namespace cuda {

void HostPermutateLoop();

void IdCopyHost2DeviceLoop();

void DeviceSampleLoop();

void GraphCopyDevice2DeviceLoop();

void IdCopyDevice2HostLoop();

void HostFeatureExtractLoop();

void FeatureCopyHost2DeviceLoop();

void SubmitLoop();

// For debug
bool RunSingleLoopOnce();
void SingleLoop();

} // namespace cuda
} // namespace common
} // namespace samgraph

#endif // SAMGRAPH_CUDA_LOOPS_H