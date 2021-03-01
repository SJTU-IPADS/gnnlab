#include <string>
#include <vector>

#include "engine.h"
#include "operation.h"
#include "logging.h"

namespace samgraph {
namespace common {

extern "C" {

void samgraph_init(const char *path, int sam_device, int train_device,
                   int batch_size, int *fanout, int num_fanout, int num_epoch) {
    SamGraphEngine::Init(path, sam_device, train_device, batch_size,
                         std::vector<int>(fanout, fanout + num_fanout),num_epoch);
    return;
}

void samgraph_start() {
    
}

void samgraph_shutdown() {
    SamGraphEngine::Shutdown();
    SAM_LOG(DEBUG) << "";
    return;
}

}

} // namespace common
} // namespace samgraph