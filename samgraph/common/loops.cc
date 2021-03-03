#include "loops.h"
#include "engine.h"

namespace samgraph {
namespace common {

bool RunHostPermutationLoopOnce() {
    return true;
}

bool RunIdCopyHost2DeviceLoopOnce() {
    return true;
}

bool RunDeviceSampleLoopOnce() {
    return true;
}

bool RunGraphCopyDevice2DeviceLoopOnce() {
    return true;
}

bool RunIdCopyDevice2HostLoopOnce() {
    return true;
}

bool RunHostFeatureSelectLoopOnce() {
    return true;
}

bool RunFeatureCopyHost2DeviceLoop() {
    return true;
}

bool RunSubmitLoopOnce() {
    return true;
}

void HostPermutationLoop() {
    while(RunHostPermutationLoopOnce() && !SamGraphEngine::ShouldShutdown()) {
    }
    SamGraphEngine::ReportThreadFinish();
}

void IdCopyHost2Device() {
    while(RunIdCopyHost2DeviceLoopOnce() && !SamGraphEngine::ShouldShutdown()) {        
    }
    SamGraphEngine::ReportThreadFinish();
}

void DeviceSampleLoop() {
    while(RunDeviceSampleLoopOnce() && !SamGraphEngine::ShouldShutdown()) {
    }
    SamGraphEngine::ReportThreadFinish();
}

void GraphCopyDevice2DeviceLoop() {
    while(RunGraphCopyDevice2DeviceLoopOnce() && !SamGraphEngine::ShouldShutdown()) {
    }
    SamGraphEngine::ReportThreadFinish();
}

void IdCopyDevice2HostLoop() {
    while(RunIdCopyDevice2HostLoopOnce() && !SamGraphEngine::ShouldShutdown()) {
    }
    SamGraphEngine::ReportThreadFinish();
}

void HostFeatureSelectLoop() {
    while(RunHostPermutationLoopOnce() && !SamGraphEngine::ShouldShutdown()) {
    }
    SamGraphEngine::ReportThreadFinish();
}

void FeatureCopyHost2DeviceLoop() {
    while(RunFeatureCopyHost2DeviceLoop() && !SamGraphEngine::ShouldShutdown()) {
    }
    SamGraphEngine::ReportThreadFinish();
}

void SubmitLoop() {
    while(RunSubmitLoopOnce() && !SamGraphEngine::ShouldShutdown()) {
    }
    SamGraphEngine::ReportThreadFinish();
}

} // namespace common
} // namespace samgraph
