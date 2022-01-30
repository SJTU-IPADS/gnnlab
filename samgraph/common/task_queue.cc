/*
 * Copyright 2022 Institute of Parallel and Distributed Systems, Shanghai Jiao Tong University
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include "task_queue.h"

#include "./dist/dist_engine.h"
#include "common.h"
#include "device.h"
#include "memory_queue.h"
#include "run_config.h"
#include "timer.h"

namespace samgraph {
namespace common {

namespace {
// TODO: hardcode mq bucket size
size_t mq_nbytes = 50 * 1024 * 1024;
} // namespace

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


// for MessageTaskQueue
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
    ret += task->output_nodes->NumBytes();
    if (!RunConfig::UseGPUCache() || RunConfig::have_switcher) {
      ret += task->input_nodes->NumBytes();
    } else {
      if (task->miss_cache_index.num_miss > 0) {
        ret += task->miss_cache_index.miss_src_index->NumBytes();
        ret += task->miss_cache_index.miss_dst_index->NumBytes();
      }

      if (task->miss_cache_index.num_cache > 0) {
        ret += task->miss_cache_index.cache_src_index->NumBytes();
        ret += task->miss_cache_index.cache_dst_index->NumBytes();
      }
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

  void CopyCPUToCPU(const void *from, void *to, size_t nbytes) {
    char *to_data = static_cast<char *>(to);
    const char *from_data = static_cast<const char *>(from);
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
    for (size_t i = 0; i < nbytes; i += 64) {
      size_t len = std::min(i + 64, nbytes);
#pragma omp simd
      for (size_t j = i; j < len; ++j) {
        to_data[j] = from_data[j];
      }
    }
  }

  void CopyGPUToCPU(const TensorPtr &from, void *to) {
    auto from_ctx = from->Ctx();
    auto stream = dist::DistEngine::Get()->GetSamplerCopyStream();
    CHECK_EQ(from_ctx.device_type, kGPU);
    Device::Get(from_ctx)->CopyDataFromTo(
        from->Data(), 0, to, 0, from->NumBytes(), from_ctx, CPU(), stream);
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
    bool have_data = false;
    if (task->graphs[0]->data != nullptr) {
      have_data = true;
    }
    size_t data_size = sizeof(TransData) + GetDataBytes(task);
    LOG(DEBUG) << "ToData transform data size: " << ToReadableSize(data_size);
    // TODO: hardcode
    CHECK_LE(data_size, mq_nbytes);

    TransData* ptr = static_cast<TransData*>(shared_ptr);
    ptr->have_data = have_data;
    ptr->num_layer = (task->graphs.size());
    ptr->key = task->key;
    ptr->input_size = task->input_nodes->Shape()[0];
    ptr->output_size = task->output_nodes->Shape()[0];
    ptr->num_miss = task->miss_cache_index.num_miss;

    IdType *ptr_data = ptr->data;

    if (!RunConfig::UseGPUCache() || RunConfig::have_switcher) {
      CHECK_EQ(sizeof(IdType) * ptr->input_size, task->input_nodes->NumBytes());
      CopyGPUToCPU(task->input_nodes, ptr_data);
      ptr_data += ptr->input_size;
    }

    CHECK_EQ(sizeof(IdType) * ptr->output_size, task->output_nodes->NumBytes());
    CopyGPUToCPU(task->output_nodes, ptr_data);
    ptr_data += ptr->output_size;

    if (RunConfig::UseGPUCache()) {
      if (ptr->num_miss > 0) {
        CopyGPUToCPU(task->miss_cache_index.miss_src_index, ptr_data);
        ptr_data += ptr->num_miss;
        CopyGPUToCPU(task->miss_cache_index.miss_dst_index, ptr_data);
        ptr_data += ptr->num_miss;
      }

      size_t num_cache = ptr->input_size - ptr->num_miss;
      if (num_cache > 0) {
        CopyGPUToCPU(task->miss_cache_index.cache_src_index, ptr_data);
        ptr_data += num_cache;
        CopyGPUToCPU(task->miss_cache_index.cache_dst_index, ptr_data);
        ptr_data += num_cache;
      }
    }

    GraphData* graph_data = reinterpret_cast<GraphData*>(ptr_data);
    LOG(DEBUG) << "ToData with task graphs layer: " << task->graphs.size();
    for (auto &graph : task->graphs) {
      graph_data->num_src = graph->num_src;
      graph_data->num_dst = graph->num_dst;
      graph_data->num_edge = graph->num_edge;

      CHECK_EQ(sizeof(IdType) * graph_data->num_edge, graph->row->NumBytes());
      CopyGPUToCPU(graph->row, graph_data->data);

      CHECK_EQ(sizeof(IdType) * graph_data->num_edge, graph->col->NumBytes());
      CopyGPUToCPU(graph->col, graph_data->data + graph_data->num_edge);

      if (have_data) {
        CHECK_EQ(sizeof(IdType) * graph_data->num_edge, graph->data->NumBytes());
        CopyGPUToCPU(graph->data, graph_data->data + 2 * graph_data->num_edge);
        graph_data = reinterpret_cast<GraphData*>(graph_data->data + 3 * graph->num_edge);
      } else {
        graph_data = reinterpret_cast<GraphData*>(graph_data->data + 2 * graph->num_edge);
      }
    }

    // sync data copy last step
    auto source_ctx = dist::DistEngine::Get()->GetSamplerCtx();
    auto stream = dist::DistEngine::Get()->GetSamplerCopyStream();
    Device::Get(source_ctx)->StreamSync(source_ctx, stream);
  }

  // cuda memory comsuption
  size_t get_cuda_used(Context ctx) {
    size_t free, used;
    CUDA_CALL(cudaSetDevice(ctx.device_id));
    CUDA_CALL(cudaMemGetInfo(&free, &used));
    return used - free;
  }

  void log_mem_usage(std::string title) {
    LOG(DEBUG) << title << "cuda usage: sampler " << ", trainer " << ToReadableSize(get_cuda_used(Engine::Get()->GetTrainerCtx())) << " with pid " << getpid();
  }

  TensorPtr ToTensor(const void* ptr, size_t nbytes, std::string name,
      Context ctx = Context{kCPU, 0}, StreamHandle stream = nullptr) {
    LOG(DEBUG) << "TaskQueue ToTensor with name: " << name;
    void *data = Device::Get(ctx)->AllocWorkspace(ctx, nbytes);
    if (ctx.device_type == kCPU) {
      CopyCPUToCPU(ptr, data, nbytes);
    } else if (ctx.device_type == kGPU) {
      Device::Get(ctx)->CopyDataFromTo(
          ptr, 0, data, 0, nbytes, CPU(), ctx, stream);
    } else {
      LOG(FATAL) << "device not supported!";
    }
    TensorPtr ret = Tensor::FromBlob(data, kI32, {(nbytes / sizeof(IdType))}, ctx, name);
    return ret;
  }

  std::shared_ptr<Task> ParseData(std::shared_ptr<SharedData> shared_data) {
    log_mem_usage("before paseData  in taskqueue");
    auto stream = dist::DistEngine::Get()->GetTrainerCopyStream();
    auto ctx = dist::DistEngine::Get()->GetTrainerCtx();
    auto trans_data = static_cast<const TransData*>(shared_data->Data());
    std::shared_ptr<Task> task = std::make_shared<Task>();
    task->key = trans_data->key;

    const IdType *trans_data_data = trans_data->data;

    if (!RunConfig::UseGPUCache() || RunConfig::have_switcher) {
      task->input_nodes =
          ToTensor(trans_data_data, trans_data->input_size * sizeof(IdType),
                   "input_" + std::to_string(task->key));
      trans_data_data += trans_data->input_size;
    }

    task->output_nodes = ToTensor(trans_data_data,
        trans_data->output_size * sizeof(IdType), "output_" + std::to_string(task->key));
    trans_data_data += trans_data->output_size;

    if (RunConfig::UseGPUCache()) {
      task->miss_cache_index.num_miss = trans_data->num_miss;
      task->miss_cache_index.num_cache =
          trans_data->input_size - trans_data->num_miss;

      if (task->miss_cache_index.num_miss > 0) {
        task->miss_cache_index.miss_src_index =
            ToTensor(trans_data_data, trans_data->num_miss * sizeof(IdType),
                     "miss_src_index_" + std::to_string(task->key));
        trans_data_data += task->miss_cache_index.num_miss;

        task->miss_cache_index.miss_dst_index =
            ToTensor(trans_data_data, trans_data->num_miss * sizeof(IdType),
                     "mis_dst_index_" + std::to_string(task->key),
                     ctx, stream);
        trans_data_data += task->miss_cache_index.num_miss;
      }

      size_t num_cache = task->miss_cache_index.num_cache;
      if (num_cache > 0) {
        task->miss_cache_index.cache_src_index =
            ToTensor(trans_data_data, num_cache * sizeof(IdType),
                     "cache_src_index_" + std::to_string(task->key),
                     ctx, stream);
        trans_data_data += num_cache;

        task->miss_cache_index.cache_dst_index =
            ToTensor(trans_data_data, num_cache * sizeof(IdType),
                     "cache_dst_index_" + std::to_string(task->key),
                     ctx, stream);
        trans_data_data += num_cache;
      }
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
          "train_graph.row_" + std::to_string(task->key) + "_" + std::to_string(layer),
          ctx, stream);
      graph->col = ToTensor(graph_data->data + graph->num_edge,
          graph->num_edge * sizeof(IdType),
          "train_graph.col_" + std::to_string(task->key) + "_" + std::to_string(layer),
          ctx, stream);
      if (trans_data->have_data) {
        graph->data = ToTensor(graph_data->data + 2 * graph->num_edge,
            graph->num_edge * sizeof(IdType),
            "train_graph.weight_" + std::to_string(task->key) + "_" + std::to_string(layer),
            ctx, stream);
        graph_data = reinterpret_cast<const GraphData*>(
            graph_data->data + 3 * graph->num_edge);
      } else {
        graph->data = nullptr;
        graph_data = reinterpret_cast<const GraphData*>(
            graph_data->data + 2 * graph->num_edge);
      }
      task->graphs[layer] = graph;
    }
    // log_mem_usage("after paseData  in taskqueue");
    // sync stream
    Device::Get(ctx)->StreamSync(ctx, stream);
    return task;
  }

  size_t GetMaxMQSize() {
    const auto &fanout = RunConfig::fanout;
    size_t layer_cnt = RunConfig::batch_size;
    size_t ret = 0;
    for (auto it = fanout.rbegin(); it != fanout.rend(); ++it) {
      size_t i = *it;
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
    if (RunConfig::have_switcher) {
      ret += (layer_cnt * sizeof(IdType));
    }
    return ret;
  }
} // namespace

MessageTaskQueue::MessageTaskQueue(size_t max_len) : TaskQueue(max_len) {
  // size_t mq_nbytes = GetMaxMQSize();
  // TODO: hardcode here to speedup the init time of FGNN
  _mq = std::make_shared<MemoryQueue>(mq_nbytes);
}

void MessageTaskQueue::Send(std::shared_ptr<Task> task) {
  size_t key;
  auto shared_data = _mq->GetPtr(key);
  ToData(task, shared_data);
  size_t bytes = sizeof(TransData) + GetDataBytes(task);
  LOG(DEBUG) << "TaskQueue Send data with " << ToReadableSize(bytes)
             << " bytes";
  _mq->SimpleSend(key);
}

std::shared_ptr<Task> MessageTaskQueue::Recv() {
  auto shared_data = _mq->Recv();
  std::shared_ptr<Task> task = ParseData(shared_data);
  LOG(DEBUG) << "TaskQueue Recv a task with key: " << task->key;
  return task;
}

}  // namespace common
}  // namespace samgraph
