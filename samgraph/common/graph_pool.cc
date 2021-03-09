#include "graph_pool.h"
#include "logging.h"

namespace samgraph {
namespace common {

GraphPool::~GraphPool() {
    {
        std::unique_lock<std::mutex> lock(_mutex);
        _stop = true;
    }
    _condition.notify_all();
}

std::shared_ptr<GraphBatch> GraphPool::GetGraphBatch(uint64_t key) {
    std::unique_lock<std::mutex> lock(_mutex);
    _condition.wait(
        lock, [this, key] { return  _stop || this->_pool.find(key) != _pool.end(); }
    );

    if (_stop && this->_pool.find(key) == _pool.end()) return nullptr;

    auto it = _pool.find(key);
    
    auto rst = it->second;
    _pool.erase(it);

    return rst;
}

void GraphPool::AddGraphBatch(uint64_t key, std::shared_ptr<GraphBatch> batch) {
    {
        std::lock_guard<std::mutex> lock(_mutex);
        SAM_CHECK(!_stop);
        SAM_CHECK_EQ(_pool.count(key), 0);
        _pool[key] = batch;
    }
    _condition.notify_one();
}

} // namespace common
} // namespace samgraph
