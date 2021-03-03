#ifndef SAMGRAPH_TASK_QUEUE_H
#define SAMGRAPH_TASK_QUEUE_H

#include <mutex>
#include <memory>

#include "common.h"

namespace samgraph {
namespace common {

class SamGraphTaskQueue {
 public:
  SamGraphTaskQueue(QueueType type);
  QueueType GetQueueType() { return _qt; }
  void AddTask(std::shared_ptr<TaskEntry>);
  std::shared_ptr<TaskEntry> GetTask();
  std::shared_ptr<TaskEntry> GetTask(uint64_t key);

 private:
  std::vector<std::shared_ptr<TaskEntry>> _q;
  std::mutex _mutex;
  QueueType _qt;
};

} // namespace common
} // namespace samgraph

#endif // SAMGRAPH_TASK_QUEUE_H

