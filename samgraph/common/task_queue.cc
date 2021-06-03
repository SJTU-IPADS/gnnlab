#include "task_queue.h"

namespace samgraph {
namespace common {

TaskQueue::TaskQueue(size_t max_len) { _max_len = max_len; }

void TaskQueue::AddTask(std::shared_ptr<Task> task) {
  std::lock_guard<std::mutex> lock(_mutex);
  _q.push_back(task);
}

size_t TaskQueue::PendingLength() {
  std::lock_guard<std::mutex> lock(_mutex);
  return _q.size();
}

bool TaskQueue::Full() {
  std::lock_guard<std::mutex> lock(_mutex);
  return _q.size() >= _max_len;
}

std::shared_ptr<Task> TaskQueue::GetTask() {
  std::lock_guard<std::mutex> lock(_mutex);
  std::shared_ptr<Task> task;
  for (auto it = _q.begin(); it != _q.end(); it++) {
    task = *it;
    _q.erase(it);

    return task;
  }
  return nullptr;
}

}  // namespace common
}  // namespace samgraph
