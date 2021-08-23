#include "task_queue.h"
#include "memory_queue.h"
#include "device.h"
#include "run_config.h"
#include "./dist/dist_engine.h"

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

  Context data_ctx = Context{kCPU, 0};
  void CopyTo(const TensorPtr& tensor, void* to) {
    auto source_ctx = tensor->Ctx();
    auto stream = dist::DistEngine::Get()->GetSamplerCopyStream();
    auto nbytes = tensor->NumBytes();
    Device::Get(source_ctx)->CopyDataFromTo(tensor->Data(), 0, to, 0,
                nbytes, source_ctx, data_ctx, stream);
    Device::Get(source_ctx)->StreamSync(source_ctx, stream);
  }

  // to print the information of a task
/*
  void PrintTask(std::shared_ptr<Task> task) {
    std::cout << "key: " << task->key <<std::endl;
    std::cout << "input size: " << task->input_nodes->Shape()[0] << std::endl;
    std::cout << "output size: " << task->output_nodes->Shape()[0] << std::endl;
    std::cout << "num layer: " << task->graphs.size() << std::endl;
    int layer = 0;
    for (const auto &g : task->graphs) {
      std::cout << "layer " << layer << ", size: " << g->num_edge << std::endl;
      ++layer;
    }
  }
*/

  void* ToData(std::shared_ptr<Task> task) {
    bool weight = false;
    if (task->graphs[0]->data != nullptr) {
      weight = true;
    }
    size_t data_size = GetDataBytes(task);
    LOG(DEBUG) << "ToData transform data size: " << data_size << " bytes";
    TransData* ptr = static_cast<TransData*>(malloc(sizeof(TransData) + data_size));
    ptr->weight = weight;
    ptr->num_layer = (task->graphs.size());
    ptr->key = task->key;
    ptr->input_size = task->input_nodes->Shape()[0];
    ptr->output_size = task->output_nodes->Shape()[0];

    CHECK_EQ(sizeof(T) * ptr->input_size, task->input_nodes->NumBytes());
    CopyTo(task->input_nodes, ptr->data);

    CHECK_EQ(sizeof(T) * ptr->output_size, task->output_nodes->NumBytes());
    CopyTo(task->output_nodes, ptr->data + ptr->input_size);

    GraphData* graph_data = reinterpret_cast<GraphData*>(ptr->data + ptr->input_size + ptr->output_size);
    LOG(DEBUG) << "ToData with task graphs layer: " << task->graphs.size();
    for (auto &graph : task->graphs) {
      graph_data->num_src = graph->num_src;
      graph_data->num_dst = graph->num_dst;
      graph_data->num_edge = graph->num_edge;

      CHECK_EQ(sizeof(T) * graph_data->num_edge, graph->row->NumBytes());
      CopyTo(graph->row, graph_data->data);

      CHECK_EQ(sizeof(T) * graph_data->num_edge, graph->col->NumBytes());
      CopyTo(graph->col, graph_data->data + graph_data->num_edge);

      if (weight) {
        CHECK_EQ(sizeof(T) * graph_data->num_edge, graph->data->NumBytes());
        CopyTo(graph->data, graph_data->data + 2 * graph_data->num_edge);
        graph_data = reinterpret_cast<GraphData*>(graph_data->data + 3 * graph->num_edge);
      } else {
        graph_data = reinterpret_cast<GraphData*>(graph_data->data + 2 * graph->num_edge);
      }
    }
    return static_cast<void*>(ptr);
  }

  TensorPtr ToTensor(const void* ptr, size_t nbytes, std::string name) {
    LOG(DEBUG) << "TaskQueue ToTensor with name: " << name;
    void* data = Device::Get(data_ctx)->
      AllocWorkspace(data_ctx, nbytes);
    std::memcpy(data, ptr, nbytes);
    TensorPtr ret = Tensor::FromBlob(data, kI32, {(nbytes/sizeof(T))}, data_ctx, name);
    return ret;
  }

  std::shared_ptr<Task> ParseData(const void* ptr) {
    auto trans_data = static_cast<const TransData*>(ptr);
    std::shared_ptr<Task> task = std::make_shared<Task>();
    task->key = trans_data->key;

    task->input_nodes = ToTensor(trans_data->data,
        trans_data->input_size * sizeof(T), "input_" + std::to_string(task->key));
    task->output_nodes = ToTensor(trans_data->data + trans_data->input_size,
        trans_data->output_size * sizeof(T), "output_" + std::to_string(task->key));

    task->graphs.resize(trans_data->num_layer);
    auto graph_data = reinterpret_cast<const GraphData*>(
        trans_data->data + trans_data->input_size + trans_data->output_size);

    int num_layer = trans_data->num_layer;
    for (int layer = 0; layer < num_layer; ++layer) {
      auto graph = std::make_shared<TrainGraph>();
      graph->num_src = graph_data->num_src;
      graph->num_dst = graph_data->num_dst;
      graph->num_edge = graph_data->num_edge;
      graph->row = ToTensor(graph_data->data,
          graph->num_edge * sizeof(T),
          "train_graph.row_" + std::to_string(task->key) + "_" + std::to_string(layer));
      graph->col = ToTensor(graph_data->data + graph->num_edge,
          graph->num_edge * sizeof(T),
          "train_graph.col_" + std::to_string(task->key) + "_" + std::to_string(layer));
      if (trans_data->weight) {
        graph->data = ToTensor(graph_data->data + 2 * graph->num_edge,
            graph->num_edge * sizeof(T),
            "train_graph.weight_" + std::to_string(task->key) + "_" + std::to_string(layer));
        graph_data = reinterpret_cast<const GraphData*>(
            graph_data->data + 3 * graph->num_edge);
      } else {
        graph->data = nullptr;
        graph_data = reinterpret_cast<const GraphData*>(
            graph_data->data + 2 * graph->num_edge);
      }
      task->graphs[layer] = graph;
    }
    return task;
  }
} // namespace

bool TaskQueue::Send(std::shared_ptr<Task> task) {
  LOG(DEBUG) << "TaskQueue Send start with task key: " << task->key;
  void* data = ToData(task);
  size_t bytes = sizeof(TransData) + GetDataBytes(task);
  LOG(DEBUG) << "TaskQueue Send data with " << bytes << " bytes";
  size_t ret = _mq->Send(data, bytes);
  return (ret == bytes);
}

std::shared_ptr<Task> TaskQueue::Recv() {
  std::shared_ptr<SharedData> shared_data = _mq->Recv();
  LOG(DEBUG) << "TaskQueue Recv data with " << shared_data->Size() << " bytes";
  std::shared_ptr<Task> task = ParseData(shared_data->Data());
  LOG(INFO) << "TaskQueue Recv a task with key: " << task->key;
  return task;
}

}  // namespace common
}  // namespace samgraph
