#include <thread>

#include "graph_pool.h"
#include "logging.h"
#include "config.h"
#include "common.h"

namespace samgraph {
namespace common {

GraphPool::~GraphPool() {
     _stop = true;
}

std::shared_ptr<GraphBatch> GraphPool::GetGraphBatch(uint64_t key) {
    uint64_t batch_key = decodeBatchKey(key);
    SAM_LOG(DEBUG) << "GraphPool: Wait for a batch with key " << batch_key;

    while(true) {
        {
            std::lock_guard<std::mutex> lock(_mutex);
            auto it = _pool.find(batch_key);
            if (this->_pool.find(batch_key) != _pool.end()) {
                SAM_LOG(DEBUG) << "GraphPool: Get batch with key " << batch_key;
                auto batch = it->second;
                _pool.erase(it);
                return batch;
            } else if (_stop) {
                return nullptr;
            }
        }
        std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    }

    return nullptr;
}

void GraphPool::AddGraphBatch(uint64_t key, std::shared_ptr<GraphBatch> batch) {
    std::lock_guard<std::mutex> lock(_mutex);
    uint64_t batch_key = decodeBatchKey(key);
    SAM_CHECK(!_stop);
    SAM_CHECK_EQ(_pool.count(batch_key), 0);
    _pool[batch_key] = batch;

    SAM_LOG(DEBUG) << "GraphPool: Add batch with key " << batch_key;
}

bool GraphPool::ExceedThreshold() {
    std::lock_guard<std::mutex> lock(_mutex);
    return _pool.size() >= _threshold;
}

} // namespace common
} // namespace samgraph
