#ifndef SAMGRAPH_TASK_QUEUE_H
#define SAMGRAPH_TASK_QUEUE_H

#include <mutex>
#include <memory>
#include <queue>

#include "common.h"

namespace samgraph {
namespace common {

class SamGraphTaskQueue {
 public:
  SamGraphTaskQueue(QueueType type, size_t threshold);
  QueueType GetQueueType() { return _qt; }
  void AddTask(std::shared_ptr<TaskEntry>);
  std::shared_ptr<TaskEntry> GetTask();
  bool ExceedThreshold();
  size_t PendingLength();

 private:
  std::queue<std::shared_ptr<TaskEntry>> _q;
  std::mutex _mutex;
  QueueType _qt;
  size_t _threshold;
};

} // namespace common
} // namespace samgraph

#endif // SAMGRAPH_TASK_QUEUE_H

