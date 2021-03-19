#include "task_queue.h"
#include "cuda/cuda_engine.h"

namespace samgraph {
namespace common {

SamGraphTaskQueue::SamGraphTaskQueue(QueueType qt, size_t threshold) {
    _qt = qt;
    _threshold = threshold;
    _rt = nullptr;

    if (_qt == SUBMIT) {
        _rt = cuda::SamGraphCudaEngine::GetSubmitTable();
    } 
}

void SamGraphTaskQueue::AddTask(std::shared_ptr<TaskEntry> task) {
    std::lock_guard<std::mutex> lock(_mutex);
    _q.push_back(task);
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
    std::shared_ptr<TaskEntry> task;
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


} // namespace common
} // namespace samgraph
