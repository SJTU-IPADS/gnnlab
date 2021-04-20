#include "cpu_loops.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <numeric>

#include "../logging.h"
#include "../profiler.h"
#include "../timer.h"
#include "cpu_engine.h"
#include "cpu_function.h"
#include "cpu_hashtable.h"

namespace samgraph {
namespace common {
namespace cpu {

bool RunCpuSampleLoopOnce() {
  auto graph_pool = CpuEngine::Get()->GetGraphPool();
  if (graph_pool->ExceedThreshold()) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    return true;
  }

  auto p = CpuEngine::Get()->GetPermutator();
  auto batch = p->GetBatch();

  if (batch) {
    // Create task entry
    Timer t;
    auto task = std::make_shared<Task>();
    task->key = CpuEngine::Get()->GetBatchKey(p->Epoch(), p->Step());
    task->output_nodes = batch;
    auto fanouts = CpuEngine::Get()->GetFanout();
    auto num_layers = fanouts.size();
    auto last_layer_idx = num_layers - 1;

    auto dataset = CpuEngine::Get()->GetGraphDataset();
    auto train_device = CpuEngine::Get()->GetTrainDevice();
    auto work_stream = CpuEngine::Get()->GetWorkStream();

    auto hash_table = CpuEngine::Get()->GetHashTable();
    hash_table->Clear();

    size_t num_train_node = task->output_nodes->shape()[0];
    hash_table->Populate(
        static_cast<const IdType *>(task->output_nodes->data()),
        num_train_node);

    task->graphs.resize(num_layers);

    const IdType *indptr = static_cast<const IdType *>(dataset->indptr->data());
    const IdType *indices =
        static_cast<const IdType *>(dataset->indices->data());

    auto cur_input = task->output_nodes;

    for (int i = last_layer_idx; i >= 0; i--) {
      Timer t0;
      const int fanout = fanouts[i];
      const IdType *input = static_cast<const IdType *>(cur_input->data());
      const size_t num_input = cur_input->shape()[0];
      LOG(DEBUG) << "CpuSample: begin sample layer " << i;

      IdType *out_src = (IdType *)malloc(num_input * fanout * sizeof(IdType));
      IdType *out_dst = (IdType *)malloc(num_input * fanout * sizeof(IdType));
      size_t num_out;
      LOG(DEBUG) << "CpuSample: cpu out_src malloc "
                 << ToReadableSize(num_input * fanout * sizeof(IdType));
      LOG(DEBUG) << "CpuSample: cpu out_src malloc "
                 << ToReadableSize(num_input * fanout * sizeof(IdType));

      // Sample a compact coo graph
      CpuSample(indptr, indices, input, num_input, out_src, out_dst, &num_out,
                fanout);
      LOG(DEBUG) << "CpuSample: num_out " << num_out;
      Profiler::Get()->num_samples[task->key] += num_out;
      double ns_time = t0.Passed();

      Timer t1;
      Timer t2;

      // Populate the hash table with newly sampled nodes
      hash_table->Populate(out_dst, num_out);

      double populate_time = t2.Passed();

      Timer t3;
      size_t num_unique = hash_table->NumItem();
      LOG(DEBUG) << "CpuSample: num_unique " << num_unique;
      IdType *unique = (IdType *)malloc(num_unique * sizeof(IdType));
      hash_table->MapNodes(unique, num_unique);

      double map_nodes_time = t3.Passed();

      Timer t4;
      // Mapping edges
      IdType *new_src = (IdType *)malloc(num_out * sizeof(IdType));
      IdType *new_dst = (IdType *)malloc(num_out * sizeof(IdType));
      LOG(DEBUG) << "CpuSample: cpu new_src malloc "
                 << ToReadableSize(num_out * sizeof(IdType));
      LOG(DEBUG) << "CpuSample: cpu new_src malloc "
                 << ToReadableSize(num_out * sizeof(IdType));
      hash_table->MapEdges(out_src, out_dst, num_out, new_src, new_dst);

      double map_edges_time = t4.Passed();

      double remap_time = t1.Passed();

      auto train_graph = std::make_shared<TrainGraph>();
      train_graph->row = Tensor::FromBlob(
          new_dst, DataType::kI32, {num_out}, CPU_DEVICE_ID,
          "train_graph.row_cpu_sample_" + std::to_string(task->key) + "_" +
              std::to_string(i));
      train_graph->col = Tensor::FromBlob(
          new_src, DataType::kI32, {num_out}, CPU_DEVICE_ID,
          "train_graph.col_cpu_sample_" + std::to_string(task->key) + "_" +
              std::to_string(i));
      train_graph->num_row = num_input;
      train_graph->num_column = num_unique;
      train_graph->num_edge = num_out;

      task->graphs[i] = train_graph;
      cur_input = Tensor::FromBlob(
          (void *)unique, DataType::kI32, {num_unique}, CPU_DEVICE_ID,
          "cur_input_unique_cpu_" + std::to_string(task->key) + "_" +
              std::to_string(i));

      free(out_src);
      free(out_dst);

      Profiler::Get()->ns_time[task->key] += ns_time;
      Profiler::Get()->remap_time[task->key] += remap_time;
      Profiler::Get()->populate_time[task->key] += populate_time;
      Profiler::Get()->map_node_time[task->key] += map_nodes_time;
      Profiler::Get()->map_edge_time[task->key] += map_edges_time;

      LOG(DEBUG) << "layer " << i << " ns " << ns_time << " remap "
                 << remap_time;

      LOG(DEBUG) << "CpuSample: finish layer " << i;
    }

    task->input_nodes = cur_input;

    // Extract feature
    auto input_nodes = task->input_nodes;
    auto output_nodes = task->output_nodes;
    CHECK_EQ(input_nodes->device(), CPU_DEVICE_ID);
    CHECK_EQ(output_nodes->device(), CPU_DEVICE_ID);

    auto feat_dim = dataset->feat->shape()[1];
    auto feat_type = dataset->feat->dtype();
    auto label_type = dataset->label->dtype();

    auto input_data = reinterpret_cast<const IdType *>(input_nodes->data());
    auto output_data = reinterpret_cast<const IdType *>(output_nodes->data());
    auto num_input = input_nodes->shape()[0];
    auto num_ouput = output_nodes->shape()[0];

    auto feat =
        Tensor::Empty(feat_type, {num_input, feat_dim}, CPU_DEVICE_ID,
                      "task.input_feat_cpu_" + std::to_string(task->key));
    auto label =
        Tensor::Empty(label_type, {num_ouput}, CPU_DEVICE_ID,
                      "task.output_label_cpu" + std::to_string(task->key));

    auto extractor = CpuEngine::Get()->GetExtractor();

    auto feat_dst = feat->mutable_data();
    auto feat_src = dataset->feat->data();
    extractor->extract(feat_dst, feat_src, input_data, num_input, feat_dim,
                       feat_type);

    auto label_dst = label->mutable_data();
    auto label_src = dataset->label->data();
    extractor->extract(label_dst, label_src, output_data, num_ouput, 1,
                       label_type);

    // Copy graph
    for (size_t i = 0; i < task->graphs.size(); i++) {
      auto graph = task->graphs[i];
      void *train_row;
      void *train_col;
      CUDA_CALL(cudaSetDevice(train_device));
      CUDA_CALL(cudaMalloc(&train_row, graph->row->size()));
      CUDA_CALL(cudaMalloc(&train_col, graph->col->size()));
      LOG(DEBUG) << "GraphCopyDevice2DeviceLoop: cuda train_row malloc "
                 << ToReadableSize(graph->row->size());
      LOG(DEBUG) << "GraphCopyDevice2DeviceLoop: cuda train_col malloc "
                 << ToReadableSize(graph->col->size());

      CUDA_CALL(cudaMemcpyAsync(train_row, graph->row->data(),
                                graph->row->size(), cudaMemcpyHostToDevice,
                                work_stream));
      CUDA_CALL(cudaStreamSynchronize(work_stream));

      CUDA_CALL(cudaMemcpyAsync(train_col, graph->col->data(),
                                graph->col->size(), cudaMemcpyHostToDevice,
                                work_stream));
      CUDA_CALL(cudaStreamSynchronize(work_stream));

      graph->row = Tensor::FromBlob(
          train_row, graph->row->dtype(), graph->row->shape(), train_device,
          "train_graph.row_cuda_train_" + std::to_string(task->key) + "_" +
              std::to_string(i));
      graph->col = Tensor::FromBlob(
          train_col, graph->col->dtype(), graph->col->shape(), train_device,
          "train_graph.col_cuda_train_" + std::to_string(task->key) + "_" +
              std::to_string(i));
    }

    CUDA_CALL(cudaStreamSynchronize(work_stream));

    // Copy data
    auto d_feat =
        Tensor::Empty(feat->dtype(), feat->shape(), train_device,
                      "task.train_feat_cuda_" + std::to_string(task->key));
    auto d_label =
        Tensor::Empty(label->dtype(), label->shape(), train_device,
                      "task.train_label_cuda" + std::to_string(task->key));

    CUDA_CALL(cudaMemcpyAsync(d_feat->mutable_data(), feat->data(),
                              feat->size(), cudaMemcpyHostToDevice,
                              work_stream));
    CUDA_CALL(cudaStreamSynchronize(work_stream));

    CUDA_CALL(cudaMemcpyAsync(d_label->mutable_data(), label->data(),
                              label->size(), cudaMemcpyHostToDevice,
                              work_stream));
    CUDA_CALL(cudaStreamSynchronize(work_stream));

    task->input_feat = d_feat;
    task->output_label = d_label;

    // Submit
    auto graph_pool = CpuEngine::Get()->GetGraphPool();
    graph_pool->AddGraphBatch(task->key, task);

    double sam_time = t.Passed();
    Profiler::Get()->sample_time[task->key] += sam_time;

    LOG(DEBUG) << "CpuSampleLoop: process task with key " << task->key;
  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }

  return true;
}

void CpuSampleLoop() {
  auto train_device = CpuEngine::Get()->GetTrainDevice();
  CUDA_CALL(cudaSetDevice(train_device));
  while (RunCpuSampleLoopOnce() && !CpuEngine::Get()->ShouldShutdown()) {
  }
  CpuEngine::Get()->ReportThreadFinish();
}

}  // namespace cpu
}  // namespace common
}  // namespace samgraph
