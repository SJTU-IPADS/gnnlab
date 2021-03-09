#ifndef SAMGRAPH_LOOPS_H
#define SAMGRAPH_LOOPS_H

namespace samgraph {
namespace common {

void HostPermutateLoop();

void IdCopyHost2DeviceLoop();

void DeviceSampleLoop();

void GraphCopyDevice2DeviceLoop();

void IdCopyDevice2HostLoop();

void HostFeatureExtractLoop();

void FeatureCopyHost2DeviceLoop();

// Submit loop will wait for the accomplishment of
// GraphCopyDevice2DeviceLoop and HostFeatureExtractLoop
void SubmitLoop();

} // namespace common
} // namespace samgraph

#endif // SAMGRAPH_LOOPS_H