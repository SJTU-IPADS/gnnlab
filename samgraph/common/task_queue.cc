#include "task_queue.h"
#include "memory_queue.h"

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

namespace {
  struct TransData {
    bool     weight; // if have weight values
    int      num_layer;
    uint64_t key;
    size_t   input_size;
    size_t   output_size;
    uint32_t data[0];
    // input_nodes|output_nodes|
    //   GraphData: <num_src|num_dst|num_edge|row|col|data> : layer 1
    //   GraphData: <num_src|num_dst|num_edge|row|col|data> : layer 2
    //   ...
  };
  struct GraphData {
    size_t num_src;
    size_t num_dst;
    size_t num_edge;
    uint32_t data[0];
    // row|col|[data]
  };
  size_t GetSize(std::shared_ptr<Task> task) {
    size_t ret = 0;
    ret += task->input_nodes->NumBytes();
    ret += task->output_nodes->NumBytes();
    for (auto &graph : task->graphs) {
      ret += 3 * sizeof(size_t);
      ret += graph->row->NumBytes();
      ret += graph->col->NumBytes();
      if (graph->data != nullptr) {
        ret += graph->data->NumBytes();
      }
    }
    return ret;
  }
  void* ToData(std::shared_ptr<Task> task) {
    bool weight = false;
    if (task->graphs[0]->data != nullptr) {
      weight = true;
    }
    size_t data_size = GetSize(task);
    TransData* ptr = malloc(sizeof(TransData) + data_size);
    ptr->weight = weight;
    ptr->num_layer = (task->graphs.size());
    ptr->key = task->key;
    ptr->input_size = task->input_nodes->Shape()[0];
    ptr->output_size = task->output_nodes->Shape()[0];
    std::memcpy(ptr->data, task->input_nodes->Data(),
                sizeof(uint32_t) * ptr->input_size);
    std::memcpy(ptr->data + ptr->input_size, task->output_nodes->Data(),
                sizeof(uint32_t) * ptr->output_size);
    GraphData* graph_data = ptr->data + ptr->input_size + ptr->output_size;
    for (auto &graph : task->graphs) {
      graph_data->num_src = graph->num_src;
      graph_data->num_dst = graph->num_dst;
      graph_data->num_edge = graph->num_edge;
      std::memcpy(graph_data->data, graph->row->Data(),
                  graph->row->NumBytes());
      std::memcpy(graph_data->data + graph->num_edge,
                  graph->col->Data(),
                  graph->col->NumBytes());
      if (weight) {
        std::memcpy(graph_data->data + 2 * graph->num_edge,
                    graph->data->Data(),
                    graph->data->NumBytes());
        graph_data = (graph_data + 3 * graph->num_edge);
      } else {
        graph_data = (graph_data + 2 * graph->num_edge);
      }
    }
    return static_cast<void*>(ptr);
  }
  std::shared_ptr<Task> ParseData(void* ptr) {

  }
} // namespace

bool TaskQueue::Send(std::shared_ptr<Task> task) {

}

}  // namespace common
}  // namespace samgraph
