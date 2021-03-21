#ifndef SAMGRAPH_TASK_QUEUE_H
#define SAMGRAPH_TASK_QUEUE_H

#include <mutex>
#include <memory>
#include <vector>

#include "common.h"
#include "ready_table.h"

namespace samgraph {
namespace common {

class SamGraphTaskQueue {
 public:
  SamGraphTaskQueue(CudaQueueType type, size_t threshold, ReadyTable* rt = nullptr);
  CudaQueueType GetQueueType() { return _qt; }
  void AddTask(std::shared_ptr<TaskEntry>);
  std::shared_ptr<TaskEntry> GetTask();
  bool ExceedThreshold();
  size_t PendingLength();

 private:
  std::vector<std::shared_ptr<TaskEntry>> _q;
  std::mutex _mutex;
  CudaQueueType _qt;
  size_t _threshold;
  ReadyTable *_rt;
};

} // namespace common
} // namespace samgraph

#endif // SAMGRAPH_TASK_QUEUE_H

