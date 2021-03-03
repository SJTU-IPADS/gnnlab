#ifndef SAMGRAPH_LOOPS_H
#define SAMGRAPH_LOOPS_H

namespace samgraph {
namespace common {

void HostPermutationLoop();

void IdCopyHost2Device();

void DeviceSampleLoop();

void GraphCopyDevice2DeviceLoop();

void IdCopyDevice2HostLoop();

void HostFeatureSelectLoop();

void FeatureCopyHost2DeviceLoop();

void SubmitLoop();

} // namespace common
} // namespace samgraph

#endif // SAMGRAPH_LOOPS_H