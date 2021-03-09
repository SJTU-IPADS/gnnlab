#include <string>
#include <vector>

#include "engine.h"
#include "operation.h"
#include "logging.h"
#include "loops.h"

namespace samgraph {
namespace common {

extern "C" {

void samgraph_init(const char *path, int sample_device, int train_device,
                   int batch_size, int *fanout, int num_fanout, int num_epoch) {
    SamGraphEngine::Init(path, sample_device, train_device,  batch_size,
                         std::vector<int>(fanout, fanout + num_fanout), num_epoch);
    return;
}

void samgraph_start() {
    std::vector<LoopFunction> func;
    func.push_back(HostPermutateLoop);
    func.push_back(IdCopyHost2DeviceLoop);
    func.push_back(DeviceSampleLoop);
    func.push_back(GraphCopyDevice2DeviceLoop);
    func.push_back(IdCopyDevice2HostLoop);
    func.push_back(HostFeatureExtractLoop);
    func.push_back(FeatureCopyHost2DeviceLoop);
    func.push_back(SubmitLoop);

    SamGraphEngine::Start(func);
}

void samgraph_shutdown() {
    SamGraphEngine::Shutdown();
    SAM_LOG(DEBUG) << "SamGraph has been completely shutdown now";
    return;
}

}

} // namespace common
} // namespace samgraph