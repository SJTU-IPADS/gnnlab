#include "graph_pool.h"

#include <thread>

#include "common.h"
#include "config.h"
#include "logging.h"

namespace samgraph {
namespace common {

GraphPool::~GraphPool() { _stop = true; }

std::shared_ptr<GraphBatch> GraphPool::GetGraphBatch(uint64_t key) {
  while (true) {
    {
      std::lock_guard<std::mutex> lock(_mutex);
      auto it = _pool.find(key);
      if (this->_pool.find(key) != _pool.end()) {
        LOG(DEBUG) << "GraphPool: Get batch with key " << key;
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
  CHECK(!_stop);
  CHECK_EQ(_pool.count(key), 0);
  _pool[key] = batch;

  LOG(DEBUG) << "GraphPool: Add batch with key " << key;
}

bool GraphPool::ExceedThreshold() {
  std::lock_guard<std::mutex> lock(_mutex);
  return _pool.size() >= _threshold;
}

}  // namespace common
}  // namespace samgraph
