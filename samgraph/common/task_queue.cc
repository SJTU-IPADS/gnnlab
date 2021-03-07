#include "task_queue.h"

namespace samgraph {
namespace common {

SamGraphTaskQueue::SamGraphTaskQueue(QueueType qt, size_t threshold) {
    _qt = qt;
    _threshold = threshold;
}

void SamGraphTaskQueue::AddTask(std::shared_ptr<TaskEntry> task) {
    std::lock_guard<std::mutex> lock(_mutex);
    _q.push(task);
}

size_t SamGraphTaskQueue::PendingLength() {
    std::lock_guard<std::mutex> lock(_mutex);
    return _q.size();
}

bool SamGraphTaskQueue::ExceedThreshold() {
    std::lock_guard<std::mutex> lock(_mutex);
    return _q.size() >= _threshold;
}

std::shared_ptr<TaskEntry> SamGraphTaskQueue::GetTask() {
    std::lock_guard<std::mutex> lock(_mutex);
    if (!_q.empty()) {
        auto task = _q.front();
        _q.pop();
        return task;
    }
    return nullptr;
}


} // namespace common
} // namespace samgraph
