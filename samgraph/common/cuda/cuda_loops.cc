#include "cuda_loops.h"

#include <cuda_runtime.h>
#include <cusparse.h>

#include <chrono>
#include <numeric>
#include <thread>

#include "../logging.h"
#include "../profiler.h"
#include "../timer.h"
#include "cuda_common.h"
#include "cuda_engine.h"
#include "cuda_function.h"
#include "cuda_hashtable.h"

namespace samgraph {
namespace common {
namespace cuda {

using TaskPtr = std::shared_ptr<Task>;

TaskPtr DoPermutate() {
  auto p = GpuEngine::Get()->GetPermutator();
  auto batch = p->GetBatch();

  if (batch) {
    auto task = std::make_shared<Task>();
    task->key = GpuEngine::Get()->GetBatchKey(p->Epoch(), p->Step());
    task->cur_input = batch;
    task->output_nodes = batch;
    LOG(DEBUG) << "DoPermutate: process task with key " << task->key;
    return task;
  } else {
    return nullptr;
  }
}

void DoGpuSample(TaskPtr task) {
  auto fanouts = GpuEngine::Get()->GetFanout();
  auto num_layers = fanouts.size();
  auto last_layer_idx = num_layers - 1;

  auto dataset = GpuEngine::Get()->GetGraphDataset();
  auto sample_device = GpuEngine::Get()->GetSampleDevice();
  auto sample_stream = GpuEngine::Get()->GetSampleStream();

  OrderedHashTable *hash_table = GpuEngine::Get()->GetHashtable();
  hash_table->Clear(sample_stream);
  CUDA_CALL(cudaStreamSynchronize(sample_stream));

  auto train_nodes = task->output_nodes;
  size_t num_train_node = train_nodes->shape()[0];
  hash_table->FillWithUnique(
      static_cast<const IdType *const>(train_nodes->data()), num_train_node,
      sample_stream);
  task->graphs.resize(num_layers);
  CUDA_CALL(cudaStreamSynchronize(sample_stream));

  const IdType *indptr = static_cast<const IdType *>(dataset->indptr->data());
  const IdType *indices = static_cast<const IdType *>(dataset->indices->data());

  for (int i = last_layer_idx; i >= 0; i--) {
    Timer t0;
    const int fanout = fanouts[i];
    const IdType *input = static_cast<const IdType *>(task->cur_input->data());
    const size_t num_input = task->cur_input->shape()[0];
    LOG(DEBUG) << "CudaSample: begin sample layer " << i;

    IdType *out_src;
    IdType *out_dst;
    size_t *num_out;
    size_t host_num_out;

    CUDA_CALL(cudaMalloc(&out_src, num_input * fanout * sizeof(IdType)));
    CUDA_CALL(cudaMalloc(&out_dst, num_input * fanout * sizeof(IdType)));
    CUDA_CALL(cudaMalloc(&num_out, sizeof(size_t)));
    LOG(DEBUG) << "CudaSample: size of out_src " << num_input * fanout;
    LOG(DEBUG) << "CudaSample: cuda out_src malloc "
               << ToReadableSize(num_input * fanout * sizeof(IdType));
    LOG(DEBUG) << "CudaSample: cuda out_dst malloc "
               << ToReadableSize(num_input * fanout * sizeof(IdType));
    LOG(DEBUG) << "CudaSample: cuda num_out malloc "
               << ToReadableSize(sizeof(size_t));

    // Sample a compact coo graph
    CudaSample((const IdType *)indptr, (const IdType *)indices,
               (const IdType *)input, (const size_t)num_input,
               (const size_t)fanout, (IdType *)out_src, (IdType *)out_dst,
               (size_t *)num_out, (cudaStream_t)sample_stream, task->key);
    // Get nnz
    CUDA_CALL(cudaMemcpyAsync(
        (void *)&host_num_out, (const void *)num_out, (size_t)sizeof(size_t),
        (cudaMemcpyKind)cudaMemcpyDeviceToHost, (cudaStream_t)sample_stream));
    CUDA_CALL(cudaStreamSynchronize(sample_stream));
    LOG(DEBUG) << "CudaSample: "
               << "layer " << i << " number of samples " << host_num_out;

    double ns_time = t0.Passed();

    Timer t1;
    Timer t2;

    // Populate the hash table with newly sampled nodes
    IdType *unique;
    size_t num_unique;

    CUDA_CALL(cudaMalloc(
        &unique, (host_num_out + hash_table->NumItems()) * sizeof(IdType)));
    LOG(DEBUG) << "CudaSample: cuda unique malloc "
               << ToReadableSize((host_num_out + +hash_table->NumItems()) *
                                 sizeof(IdType));

    hash_table->FillWithDuplicates(out_dst, host_num_out, unique, &num_unique,
                                   sample_stream);

    double populate_time = t2.Passed();

    Timer t3;

    // Mapping edges
    IdType *new_src;
    IdType *new_dst;

    CUDA_CALL(cudaMalloc(&new_src, host_num_out * sizeof(IdType)));
    CUDA_CALL(cudaMalloc(&new_dst, host_num_out * sizeof(IdType)));
    LOG(DEBUG) << "CudaSample: size of new_src " << host_num_out;
    LOG(DEBUG) << "CudaSample: cuda new_src malloc "
               << ToReadableSize(host_num_out * sizeof(IdType));
    LOG(DEBUG) << "CudaSample: cuda new_dst malloc "
               << ToReadableSize(host_num_out * sizeof(IdType));

    MapEdges((const IdType *)out_src, (IdType *const)new_src,
             (const IdType *const)out_dst, (IdType *const)new_dst,
             (const size_t)host_num_out,
             (DeviceOrderedHashTable)hash_table->DeviceHandle(),
             (cudaStream_t)sample_stream);

    double map_edges_time = t3.Passed();
    double remap_time = t1.Passed();

    auto train_graph = std::make_shared<TrainGraph>();
    train_graph->num_row = num_unique;
    train_graph->num_column = num_input;
    train_graph->num_edge = host_num_out;
    train_graph->col = Tensor::FromBlob(
        new_src, DataType::kI32, {host_num_out}, sample_device,
        "train_graph.row_cuda_sample_" + std::to_string(task->key) + "_" +
            std::to_string(i));
    train_graph->row = Tensor::FromBlob(
        new_dst, DataType::kI32, {host_num_out}, sample_device,
        "train_graph.dst_cuda_sample_" + std::to_string(task->key) + "_" +
            std::to_string(i));

    task->graphs[i] = train_graph;

    // Do some clean jobs
    CUDA_CALL(cudaFree(out_src));
    CUDA_CALL(cudaFree(out_dst));
    CUDA_CALL(cudaFree(num_out));

    LOG(DEBUG) << "layer " << i << " ns " << ns_time << " remap " << remap_time;

    Profiler::Get()->num_samples[task->key] += host_num_out;
    Profiler::Get()->ns_time[task->key] += ns_time;
    Profiler::Get()->remap_time[task->key] += remap_time;
    Profiler::Get()->populate_time[task->key] += populate_time;
    Profiler::Get()->map_node_time[task->key] += 0;
    Profiler::Get()->map_edge_time[task->key] += map_edges_time;

    task->cur_input = Tensor::FromBlob(
        (void *)unique, DataType::kI32, {num_unique}, sample_device,
        "cur_input_unique_cuda_" + std::to_string(task->key) + "_" +
            std::to_string(i));
    LOG(DEBUG) << "CudaSample: finish layer " << i;
  }

  task->input_nodes = task->cur_input;

  LOG(DEBUG) << "SampleLoop: process task with key " << task->key;
}

void DoGraphCopy(TaskPtr task) {
  auto train_device = GpuEngine::Get()->GetTrainDevice();
  auto copy_stream = GpuEngine::Get()->GetCopyStream();

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

    CUDA_CALL(cudaMemcpyAsync(train_row, graph->row->data(), graph->row->size(),
                              cudaMemcpyDeviceToDevice, copy_stream));
    CUDA_CALL(cudaStreamSynchronize(copy_stream));

    CUDA_CALL(cudaMemcpyAsync(train_col, graph->col->data(), graph->col->size(),
                              cudaMemcpyDeviceToDevice, copy_stream));
    CUDA_CALL(cudaStreamSynchronize(copy_stream));

    graph->row = Tensor::FromBlob(
        train_row, graph->row->dtype(), graph->row->shape(), train_device,
        "train_graph.row_cuda_train_" + std::to_string(task->key) + "_" +
            std::to_string(i));
    graph->col = Tensor::FromBlob(
        train_col, graph->col->dtype(), graph->col->shape(), train_device,
        "train_graph.col_cuda_train_" + std::to_string(task->key) + "_" +
            std::to_string(i));
  }

  CUDA_CALL(cudaStreamSynchronize(copy_stream));

  LOG(DEBUG) << "GraphCopyDevice2Device: process task with key " << task->key;
}

void DoIdCopy(TaskPtr task) {
  auto copy_stream = GpuEngine::Get()->GetCopyStream();

  void *input_nodes = malloc(task->input_nodes->size());
  void *output_nodes = malloc(task->output_nodes->size());
  LOG(DEBUG) << "IdCopyDevice2Host input_nodes cpu malloc "
             << ToReadableSize(task->input_nodes->size());
  LOG(DEBUG) << "IdCopyDevice2Host output_nodes cpu malloc "
             << ToReadableSize(task->output_nodes->size());

  CUDA_CALL(cudaMemcpyAsync(input_nodes, task->input_nodes->data(),
                            task->input_nodes->size(), cudaMemcpyDeviceToHost,
                            copy_stream));
  CUDA_CALL(cudaMemcpyAsync(output_nodes, task->output_nodes->data(),
                            task->output_nodes->size(), cudaMemcpyDeviceToHost,
                            copy_stream));
  CUDA_CALL(cudaStreamSynchronize(copy_stream));

  task->input_nodes = Tensor::FromBlob(
      input_nodes, task->input_nodes->dtype(), task->input_nodes->shape(),
      CPU_DEVICE_ID, "task.input_nodes_cpu_" + std::to_string(task->key));
  task->output_nodes = Tensor::FromBlob(
      output_nodes, task->output_nodes->dtype(), task->output_nodes->shape(),
      CPU_DEVICE_ID, "task.output_nodes_cpu_" + std::to_string(task->key));

  LOG(DEBUG) << "IdCopyDevice2Host: process task with key " << task->key;
}

void DoFeatureExtract(TaskPtr task) {
  auto dataset = GpuEngine::Get()->GetGraphDataset();
  auto input_nodes = task->input_nodes;
  auto output_nodes = task->output_nodes;
  CHECK_EQ(input_nodes->device(), CPU_DEVICE_ID);
  CHECK_EQ(output_nodes->device(), CPU_DEVICE_ID);

  auto feat = dataset->feat;
  auto label = dataset->label;

  auto feat_dim = dataset->feat->shape()[1];
  auto feat_type = dataset->feat->dtype();
  auto label_type = dataset->label->dtype();

  auto input_data = reinterpret_cast<const IdType *>(input_nodes->data());
  auto output_data = reinterpret_cast<const IdType *>(output_nodes->data());
  auto num_input = input_nodes->shape()[0];
  auto num_ouput = output_nodes->shape()[0];

  task->input_feat =
      Tensor::Empty(feat_type, {num_input, feat_dim}, CPU_DEVICE_ID,
                    "task.input_feat_cpu_" + std::to_string(task->key));
  task->output_label =
      Tensor::Empty(label_type, {num_ouput}, CPU_DEVICE_ID,
                    "task.output_label_cpu" + std::to_string(task->key));

  auto extractor = GpuEngine::Get()->GetExtractor();

  auto feat_dst = task->input_feat->mutable_data();
  auto feat_src = dataset->feat->data();
  extractor->extract(feat_dst, feat_src, input_data, num_input, feat_dim,
                     feat_type);

  auto label_dst = task->output_label->mutable_data();
  auto label_src = dataset->label->data();
  extractor->extract(label_dst, label_src, output_data, num_ouput, 1,
                     label_type);

  LOG(DEBUG) << "HostFeatureExtract: process task with key " << task->key;
}

void DoFeatureCopy(TaskPtr task) {
  auto train_device = GpuEngine::Get()->GetTrainDevice();
  CUDA_CALL(cudaSetDevice(train_device));
  auto copy_stream = GpuEngine::Get()->GetCopyStream();

  auto feat = task->input_feat;
  auto label = task->output_label;

  auto d_feat =
      Tensor::Empty(feat->dtype(), feat->shape(), train_device,
                    "task.train_feat_cuda_" + std::to_string(task->key));
  auto d_label =
      Tensor::Empty(label->dtype(), label->shape(), train_device,
                    "task.train_label_cuda" + std::to_string(task->key));

  CUDA_CALL(cudaMemcpyAsync(d_feat->mutable_data(), feat->data(), feat->size(),
                            cudaMemcpyHostToDevice, copy_stream));
  CUDA_CALL(cudaStreamSynchronize(copy_stream));

  CUDA_CALL(cudaMemcpyAsync(d_label->mutable_data(), label->data(),
                            label->size(), cudaMemcpyHostToDevice,
                            copy_stream));
  CUDA_CALL(cudaStreamSynchronize(copy_stream));

  task->input_feat = d_feat;
  task->output_label = d_label;

  LOG(DEBUG) << "FeatureCopyHost2Device: process task with key " << task->key;
}

bool RunGpuSampleLoopOnce() {
  auto next_op = kDataCopy;
  auto next_q = GpuEngine::Get()->GetTaskQueue(next_op);
  if (next_q->ExceedThreshold()) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    return true;
  }

  Timer t0;
  auto task = DoPermutate();
  if (!task) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    return true;
  }
  double shuffle_time = t0.Passed();

  Timer t1;
  DoGpuSample(task);
  double sample_time = t1.Passed();

  next_q->AddTask(task);

  Profiler::Get()->sample_time[task->key] = shuffle_time + sample_time;
  Profiler::Get()->shuffle_time[task->key] = shuffle_time;
  Profiler::Get()->real_sample_time[task->key] = sample_time;

  return true;
}

bool RunDataCopyLoopOnce() {
  auto graph_pool = GpuEngine::Get()->GetGraphPool();
  if (graph_pool->ExceedThreshold()) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    return true;
  }

  auto this_op = kDataCopy;
  auto q = GpuEngine::Get()->GetTaskQueue(this_op);
  auto task = q->GetTask();

  if (task) {
    Timer t0;
    DoGraphCopy(task);
    double graph_copy_time = t0.Passed();

    Timer t1;
    DoIdCopy(task);
    double id_copy_time = t1.Passed();

    Timer t2;
    DoFeatureExtract(task);
    double extract_time = t2.Passed();

    Timer t3;
    DoFeatureCopy(task);
    double feat_copy_time = t3.Passed();

    LOG(DEBUG) << "Submit: process task with key " << task->key;
    graph_pool->AddGraphBatch(task->key, task);

    Profiler::Get()->copy_time[task->key] =
        graph_copy_time + id_copy_time + extract_time + feat_copy_time;
    Profiler::Get()->graph_copy_time[task->key] = graph_copy_time;
    Profiler::Get()->id_copy_time[task->key] = id_copy_time;
    Profiler::Get()->extract_time[task->key] = extract_time;
    Profiler::Get()->feat_copy_time[task->key] = feat_copy_time;

  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }

  return true;
}

void GpuSampleLoop() {
  while (RunGpuSampleLoopOnce() && !GpuEngine::Get()->ShouldShutdown()) {
  }
  GpuEngine::Get()->ReportThreadFinish();
}

void DataCopyLoop() {
  while (RunDataCopyLoopOnce() && !GpuEngine::Get()->ShouldShutdown()) {
  }
  GpuEngine::Get()->ReportThreadFinish();
}

}  // namespace cuda
}  // namespace common
}  // namespace samgraph
