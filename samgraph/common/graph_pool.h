#ifndef SAMGRAPH_GRAPH_POOL_H
#define SAMGRAPH_GRAPH_POOL_H

#include <memory>
#include <mutex>
#include <unordered_map>
#include <queue>

#include "common.h"

namespace samgraph {
namespace common {

class GraphPool {
 public:
  GraphPool(size_t max_size) : _stop(false), _max_size(max_size) {}
  ~GraphPool();

  std::shared_ptr<GraphBatch> GetGraphBatch();
  void Submit(uint64_t key, std::shared_ptr<GraphBatch> batch);
  bool Full();

 private:
  bool _stop;
  std::mutex _mutex;
  const size_t _max_size;
  // std::unordered_map<uint64_t, std::shared_ptr<GraphBatch>> _pool;
  std::queue<std::shared_ptr<GraphBatch>> _pool;
};

}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_GRAPH_POLL_H
