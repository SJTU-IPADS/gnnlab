#ifndef SAMGRAPH_CUDA_LOOPS_H
#define SAMGRAPH_CUDA_LOOPS_H

namespace samgraph {
namespace common {
namespace cuda {

void HostPermutateLoop();

void CudaSample();

void GraphCopyDevice2DeviceLoop();

void IdCopyDevice2HostLoop();

void HostFeatureExtractLoop();

void FeatureCopyHost2DeviceLoop();

void SubmitLoop();

void SingleLoop();

// For debug
bool RunHostPermutateLoopOnce();
bool RunCudaSampleLoopOnce();
bool RunSingleLoopOnce();

} // namespace cuda
} // namespace common
} // namespace samgraph

#endif // SAMGRAPH_CUDA_LOOPS_H