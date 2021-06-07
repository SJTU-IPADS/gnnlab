#ifndef SAMGRAPH_TASK_QUEUE_H
#define SAMGRAPH_TASK_QUEUE_H

#include <memory>
#include <mutex>
#include <vector>

#include "common.h"

namespace samgraph {
namespace common {

class TaskQueue {
 public:
  TaskQueue(size_t max_len);

  void AddTask(std::shared_ptr<Task>);
  std::shared_ptr<Task> GetTask();
  bool Full();
  size_t PendingLength();

 private:
  std::vector<std::shared_ptr<Task>> _q;
  std::mutex _mutex;
  size_t _max_len;
};

}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_TASK_QUEUE_H
