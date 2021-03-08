#include "graph_pool.h"
#include "logging.h"

namespace samgraph {
namespace common {

std::shared_ptr<GraphBatch> GraphPool::GetGraphBatch(uint64_t key) {
    std::lock_guard<std::mutex> lock(_mutex);
    auto it = _pool.find(key);
    if (it == _pool.end()) {
        return nullptr;
    } else {
        return it->second;
    }
}

void GraphPool::AddGraphBatch(uint64_t key, std::shared_ptr<GraphBatch> batch) {
    std::lock_guard<std::mutex> lock(_mutex);
    SAM_CHECK_EQ(_pool.count(key), 0);
    _pool[key] = batch;
}

} // namespace common
} // namespace samgraph
