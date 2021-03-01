#ifndef SAMGRAPH_LOOPS_H
#define SAMGRAPH_LOOPS_H

namespace samgraph {
namespace common {

void HostShuffleLoop();

void IdCopyHost2Device();

void DeviceSampleLoop();

void GraphCopyDevice2DeviceLoop();

void IdCopyDevice2HostLoop();

void FeatureHostIndexSelectLoop();

void FeatureCopyHost2DeviceLoop();

} // namespace common
} // namespace samgraph

#endif // SAMGRAPH_LOOPS_H