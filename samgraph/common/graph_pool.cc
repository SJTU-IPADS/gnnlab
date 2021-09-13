#include "graph_pool.h"

#include <thread>

#include "common.h"
#include "constant.h"
#include "logging.h"

namespace samgraph {
namespace common {

GraphPool::~GraphPool() { _stop = true; }

std::shared_ptr<GraphBatch> GraphPool::GetGraphBatch() {
  while (true) {
    {
      std::lock_guard<std::mutex> lock(_mutex);
      if (!_pool.empty()) {
        auto batch = _pool.front();
        _pool.pop();
        auto key = batch->key;
        LOG(DEBUG) << "GraphPool: Get batch with key " << key;
        return batch;
      } else if (_stop) {
        return nullptr;
      }
    }
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }

  return nullptr;
}

void GraphPool::Submit(uint64_t key, std::shared_ptr<GraphBatch> batch) {
  std::lock_guard<std::mutex> lock(_mutex);
  CHECK(!_stop);
  _pool.push(batch);

  LOG(DEBUG) << "GraphPool: Add batch with key " << key;
}

bool GraphPool::Full() {
  std::lock_guard<std::mutex> lock(_mutex);
  return _pool.size() >= _max_size;
}

}  // namespace common
}  // namespace samgraph
