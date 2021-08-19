#include "task_queue.h"
#include "memory_queue.h"
#include "device.h"
#include "run_config.h"

namespace samgraph {
namespace common {

TaskQueue::TaskQueue(size_t max_len) {
  _max_len = max_len;
  _mq = std::make_shared<MemoryQueue>(RunConfig::shared_meta_path);
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
  using T = uint32_t;

  struct TransData {
    bool     weight; // if have weight values
    int      num_layer;
    uint64_t key;
    size_t   input_size;
    size_t   output_size;
    T data[0];
    // input_nodes|output_nodes|
    //   GraphData: <num_src|num_dst|num_edge|row|col|data> : layer 1
    //   GraphData: <num_src|num_dst|num_edge|row|col|data> : layer 2
    //   ...
  };

  struct GraphData {
    size_t num_src;
    size_t num_dst;
    size_t num_edge;
    T data[0];
    // row|col|[data]
  };

  size_t GetDataBytes(std::shared_ptr<Task> task) {
    size_t ret = 0;
    ret += task->input_nodes->NumBytes();
    ret += task->output_nodes->NumBytes();
    for (auto &graph : task->graphs) {
      ret += sizeof(GraphData);
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
    size_t data_size = GetDataBytes(task);
    TransData* ptr = static_cast<TransData*>(malloc(sizeof(TransData) + data_size));
    ptr->weight = weight;
    ptr->num_layer = (task->graphs.size());
    ptr->key = task->key;
    ptr->input_size = task->input_nodes->Shape()[0];
    ptr->output_size = task->output_nodes->Shape()[0];

    CHECK_EQ(sizeof(T) * ptr->input_size, task->input_nodes->NumBytes());
    std::memcpy(ptr->data, task->input_nodes->Data(),
                sizeof(T) * ptr->input_size);

    CHECK_EQ(sizeof(T) * ptr->output_size, task->output_nodes->NumBytes());
    std::memcpy(ptr->data + ptr->input_size, task->output_nodes->Data(),
                sizeof(T) * ptr->output_size);

    GraphData* graph_data = reinterpret_cast<GraphData*>(ptr->data + ptr->input_size + ptr->output_size);
    for (auto &graph : task->graphs) {
      graph_data->num_src = graph->num_src;
      graph_data->num_dst = graph->num_dst;
      graph_data->num_edge = graph->num_edge;
      CHECK_EQ(sizeof(T) * graph_data->num_edge, graph->row->NumBytes());
      std::memcpy(graph_data->data, graph->row->Data(),
                  graph->row->NumBytes());
      CHECK_EQ(sizeof(T) * graph_data->num_edge, graph->col->NumBytes());
      std::memcpy(graph_data->data + graph->num_edge,
                  graph->col->Data(),
                  graph->col->NumBytes());
      if (weight) {
        CHECK_EQ(sizeof(T) * graph_data->num_edge, graph->data->NumBytes());
        std::memcpy(graph_data->data + 2 * graph->num_edge,
                    graph->data->Data(),
                    graph->data->NumBytes());
        graph_data = reinterpret_cast<GraphData*>(graph_data->data + 3 * graph->num_edge);
      } else {
        graph_data = reinterpret_cast<GraphData*>(graph_data->data + 2 * graph->num_edge);
      }
    }
    return static_cast<void*>(ptr);
  }

  Context data_ctx = Context{kCPU, 0};
  TensorPtr to_tensor(const void* ptr, size_t nbytes, std::string name) {
    void* data = Device::Get(data_ctx)->
      AllocWorkspace(data_ctx, nbytes);
    std::memcpy(data, ptr, nbytes);
    TensorPtr ret = Tensor::FromBlob(data, kI32, {(nbytes/sizeof(T))}, data_ctx, name);
    return ret;
  }

  std::shared_ptr<Task> ParseData(const void* ptr) {
    const TransData* trans_data = static_cast<const TransData*>(ptr);
    std::shared_ptr<Task> task = std::make_shared<Task>();
    task->key = trans_data->key;

    task->input_nodes = to_tensor(trans_data->data,
        trans_data->input_size * sizeof(T), "input_" + std::to_string(task->key));
    task->output_nodes = to_tensor(trans_data->data + trans_data->input_size,
        trans_data->output_size * sizeof(T), "output_" + std::to_string(task->key));

    task->graphs.resize(trans_data->num_layer);
    const GraphData *graph_data = reinterpret_cast<const GraphData*>(
        trans_data->data + trans_data->input_size + trans_data->output_size);

    int layer = 0;
    for (auto &graph : task->graphs) {
      graph->num_src = graph_data->num_src;
      graph->num_dst = graph_data->num_dst;
      graph->num_edge = graph_data->num_edge;
      graph->row = to_tensor(graph_data->data,
          graph->num_edge * sizeof(T),
          "train_graph.row_" + std::to_string(task->key) + "_" + std::to_string(layer));
      graph->col = to_tensor(graph_data->data + graph->num_edge,
          graph->num_edge * sizeof(T),
          "train_graph.col_" + std::to_string(task->key) + "_" + std::to_string(layer));
      if (trans_data->weight) {
        graph->data = to_tensor(graph_data->data + 2 * graph->num_edge,
            graph->num_edge * sizeof(T),
            "train_graph.weight_" + std::to_string(task->key) + "_" + std::to_string(layer));
        graph_data = reinterpret_cast<const GraphData*>(
            graph_data->data + 3 * graph->num_edge);
      } else {
        graph->data = nullptr;
        graph_data = reinterpret_cast<const GraphData*>(
            graph_data->data + 2 * graph->num_edge);
      }
      ++layer;
    }
    return task;
  }
} // namespace

bool TaskQueue::Send(std::shared_ptr<Task> task) {
  void* data = ToData(task);
  size_t bytes = sizeof(TransData) + GetDataBytes(task);
  size_t ret = _mq->Send(data, bytes);
  return (ret == bytes);
}

std::shared_ptr<Task> TaskQueue::Recv() {
  std::shared_ptr<SharedData> shared_data = _mq->Recv();
  std::shared_ptr<Task> task = ParseData(shared_data->Data());
  return task;
}

}  // namespace common
}  // namespace samgraph
