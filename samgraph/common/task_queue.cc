#include "task_queue.h"
#include "memory_queue.h"
#include "device.h"
#include "run_config.h"
#include "./dist/dist_engine.h"
#include "timer.h"

namespace samgraph {
namespace common {

namespace {

  struct TransData {
    bool     have_data; // if have weight values
    int      num_layer;
    uint64_t key;
    size_t   input_size;
    size_t   output_size;
    size_t   num_miss;
    IdType data[0];
    // input_nodes|output_nodes| [input_dst_index]
    //   GraphData: <num_src|num_dst|num_edge|row|col|data> : layer 1
    //   GraphData: <num_src|num_dst|num_edge|row|col|data> : layer 2
    //   ...
  };

  struct GraphData {
    size_t num_src;
    size_t num_dst;
    size_t num_edge;
    IdType data[0];
    // row|col|[data]
  };

  size_t GetDataBytes(std::shared_ptr<Task> task) {
    size_t ret = 0;
    ret += task->input_nodes->NumBytes();
    ret += task->output_nodes->NumBytes();
    if (task->input_dst_index != nullptr) {
      ret += task->input_dst_index->NumBytes();
    }
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

  void ToData(std::shared_ptr<Task> task, void* shared_ptr) {
    // Timer tt;
    bool have_data = false;
    if (task->graphs[0]->data != nullptr) {
      have_data = true;
    }
    size_t data_size = GetDataBytes(task);
    LOG(DEBUG) << "ToData transform data size: " << data_size << " bytes";
    TransData* ptr = static_cast<TransData*>(shared_ptr);
    ptr->have_data = have_data;
    ptr->num_layer = (task->graphs.size());
    ptr->key = task->key;
    ptr->input_size = task->input_nodes->Shape()[0];
    ptr->output_size = task->output_nodes->Shape()[0];
    ptr->num_miss = task->num_miss;

    // Timer t0;
    IdType* ptr_data = ptr->data;
    CHECK_EQ(sizeof(IdType) * ptr->input_size, task->input_nodes->NumBytes());
    CopyTo(task->input_nodes, ptr_data);
    ptr_data += ptr->input_size;

    CHECK_EQ(sizeof(IdType) * ptr->output_size, task->output_nodes->NumBytes());
    CopyTo(task->output_nodes, ptr_data);
    ptr_data += ptr->output_size;

    if (task->input_dst_index != nullptr) {
      CHECK_NE(ptr->num_miss, ptr->input_size);
      CopyTo(task->input_dst_index, ptr_data);
      ptr_data += ptr->input_size;
    }
    // double node_time = t0.Passed();

    // Timer t1;
    GraphData* graph_data = reinterpret_cast<GraphData*>(ptr_data);
    LOG(DEBUG) << "ToData with task graphs layer: " << task->graphs.size();
    for (auto &graph : task->graphs) {
      graph_data->num_src = graph->num_src;
      graph_data->num_dst = graph->num_dst;
      graph_data->num_edge = graph->num_edge;

      CHECK_EQ(sizeof(IdType) * graph_data->num_edge, graph->row->NumBytes());
      CopyTo(graph->row, graph_data->data);

      CHECK_EQ(sizeof(IdType) * graph_data->num_edge, graph->col->NumBytes());
      CopyTo(graph->col, graph_data->data + graph_data->num_edge);

      if (have_data) {
        CHECK_EQ(sizeof(IdType) * graph_data->num_edge, graph->data->NumBytes());
        CopyTo(graph->data, graph_data->data + 2 * graph_data->num_edge);
        graph_data = reinterpret_cast<GraphData*>(graph_data->data + 3 * graph->num_edge);
      } else {
        graph_data = reinterpret_cast<GraphData*>(graph_data->data + 2 * graph->num_edge);
      }
    }
    // double graph_time = t1.Passed();
    // double total_time = tt.Passed();
    // printf("copy nodes: %.6f, copy graph: %.6f, total: %.6f\n", node_time, graph_time, total_time);
  }

  TensorPtr ToTensor(const void* ptr, size_t nbytes, std::string name) {
    LOG(DEBUG) << "TaskQueue ToTensor with name: " << name;
    void* data = Device::Get(data_ctx)->
      AllocWorkspace(data_ctx, nbytes);
    Device::Get(data_ctx)->CopyDataFromTo(ptr, 0, data, 0, nbytes, data_ctx, data_ctx);
    TensorPtr ret = Tensor::FromBlob(data, kI32, {(nbytes/sizeof(IdType))}, data_ctx, name);
    return ret;
  }

  std::shared_ptr<Task> ParseData(const void* ptr) {
    auto trans_data = static_cast<const TransData*>(ptr);
    std::shared_ptr<Task> task = std::make_shared<Task>();
    task->key = trans_data->key;
    task->num_miss = trans_data->num_miss;

    const IdType *trans_data_data = trans_data->data;

    task->input_nodes = ToTensor(trans_data_data,
        trans_data->input_size * sizeof(IdType), "input_" + std::to_string(task->key));
    trans_data_data += trans_data->input_size;

    task->output_nodes = ToTensor(trans_data_data,
        trans_data->output_size * sizeof(IdType), "output_" + std::to_string(task->key));
    trans_data_data += trans_data->output_size;

    if (RunConfig::UseGPUCache()) {
      task->input_dst_index = ToTensor(trans_data_data,
          trans_data->input_size * sizeof(IdType), "cache_dst_index_" + std::to_string(task->key));
      trans_data_data += trans_data->input_size;
    }

    task->graphs.resize(trans_data->num_layer);
    auto graph_data = reinterpret_cast<const GraphData*>(trans_data_data);

    int num_layer = trans_data->num_layer;
    for (int layer = 0; layer < num_layer; ++layer) {
      auto graph = std::make_shared<TrainGraph>();
      graph->num_src = graph_data->num_src;
      graph->num_dst = graph_data->num_dst;
      graph->num_edge = graph_data->num_edge;
      graph->row = ToTensor(graph_data->data,
          graph->num_edge * sizeof(IdType),
          "train_graph.row_" + std::to_string(task->key) + "_" + std::to_string(layer));
      graph->col = ToTensor(graph_data->data + graph->num_edge,
          graph->num_edge * sizeof(IdType),
          "train_graph.col_" + std::to_string(task->key) + "_" + std::to_string(layer));
      if (trans_data->have_data) {
        graph->data = ToTensor(graph_data->data + 2 * graph->num_edge,
            graph->num_edge * sizeof(IdType),
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

  size_t GetMaxMQSize() {
    const auto &fanout = RunConfig::fanout;
    size_t layer_cnt = RunConfig::batch_size;
    size_t ret = 0;
    for (auto i : fanout) {
      ret += sizeof(GraphData);
      ret += (layer_cnt * i * 3 * sizeof(IdType));
      layer_cnt = layer_cnt + layer_cnt * i;
    }
    ret += sizeof(TransData);
    ret += (RunConfig::batch_size * sizeof(IdType)); // label size;
    ret += (layer_cnt * sizeof(IdType));
    if (RunConfig::UseGPUCache()) {
      ret += (layer_cnt * sizeof(IdType));
    }
    return ret;
  }
} // namespace

TaskQueue::TaskQueue(size_t max_len) {
  _max_len = max_len;
  size_t mq_nbytes = GetMaxMQSize();
  _mq = std::make_shared<MemoryQueue>(RunConfig::shared_meta_path, mq_nbytes);
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


bool TaskQueue::Send(std::shared_ptr<Task> task) {
  // Timer tt;
  // Timer t0;
  LOG(DEBUG) << "TaskQueue Send start with task key: " << task->key;
  size_t key;
  void* shared_ptr = _mq->GetPos(key);
  ToData(task, shared_ptr);
  size_t bytes = sizeof(TransData) + GetDataBytes(task);
  LOG(DEBUG) << "TaskQueue Send data with " << bytes << " bytes";
  // double send_time = t0.Passed();
  // size_t ret = _mq->Send(data, bytes);
  _mq->SimpleSend(key);
  // double total = tt.Passed();
  return true;
}

std::shared_ptr<Task> TaskQueue::Recv() {
  void* shared_data = _mq->Recv();
  std::shared_ptr<Task> task = ParseData(shared_data);
  LOG(INFO) << "TaskQueue Recv a task with key: " << task->key;
  return task;
}

}  // namespace common
}  // namespace samgraph
