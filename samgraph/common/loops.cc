#include "loops.h"

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
