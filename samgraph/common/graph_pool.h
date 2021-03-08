#ifndef SAMGRAPH_GRAPH_POOL_H
#define SAMGRAPH_GRAPH_POOL_H

#include <unordered_map>
#include <memory>
#include <mutex>

#include "common.h"

namespace samgraph {
namespace common {

class GraphPool {
 public:
  std::shared_ptr<GraphBatch> GetGraphBatch(uint64_t key);
  void AddGraphBatch(uint64_t key, std::shared_ptr<GraphBatch> batch);
 private:
  std::mutex _mutex;
  std::unordered_map<uint64_t, std::shared_ptr<GraphBatch>> _pool;
};

} // namespace common
} // namespace samgraph

#endif // SAMGRAPH_GRAPH_POLL_H