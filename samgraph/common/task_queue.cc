#include "task_queue.h"

namespace samgraph {
namespace common {

TaskQueue::TaskQueue(size_t threshold, ReadyTable* rt) {
  _threshold = threshold;
  _rt = rt;
}

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
  return _q.size() >= _threshold;
}

std::shared_ptr<Task> TaskQueue::GetTask() {
  std::lock_guard<std::mutex> lock(_mutex);
  std::shared_ptr<Task> task;
  for (auto it = _q.begin(); it != _q.end(); it++) {
    if (_rt) {
      if (!_rt->IsKeyReady((*it)->key)) {
        continue;
      }
    }

    task = *it;
    _q.erase(it);

    return task;
  }
  return nullptr;
}

}  // namespace common
}  // namespace samgraph
