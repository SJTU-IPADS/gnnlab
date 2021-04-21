#include "cuda_loops.h"

#include <cuda_runtime.h>
#include <cusparse.h>

#include <chrono>
#include <numeric>
#include <thread>

#include "../device.h"
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
  auto sampler_ctx = GpuEngine::Get()->GetSamplerCtx();
  auto sampler_device = Device::Get(sampler_ctx);
  auto sample_stream = GpuEngine::Get()->GetSampleStream();

  OrderedHashTable *hash_table = GpuEngine::Get()->GetHashtable();
  hash_table->Reset(sample_stream);

  auto output_nodes = task->output_nodes;
  size_t num_train_node = output_nodes->Shape()[0];
  hash_table->FillWithUnique(
      static_cast<const IdType *const>(output_nodes->Data()), num_train_node,
      sample_stream);
  task->graphs.resize(num_layers);

  const IdType *indptr = static_cast<const IdType *>(dataset->indptr->Data());
  const IdType *indices = static_cast<const IdType *>(dataset->indices->Data());

  auto cur_input = task->output_nodes;

  for (int i = last_layer_idx; i >= 0; i--) {
    Timer t0;
    const int fanout = fanouts[i];
    const IdType *input = static_cast<const IdType *>(cur_input->Data());
    const size_t num_input = cur_input->Shape()[0];
    LOG(DEBUG) << "DoGpuSample: begin sample layer " << i;

    IdType *out_src = static_cast<IdType *>(sampler_device->AllocWorkspace(
        sampler_ctx, num_input * fanout * sizeof(IdType)));
    IdType *out_dst = static_cast<IdType *>(sampler_device->AllocWorkspace(
        sampler_ctx, num_input * fanout * sizeof(IdType)));
    size_t *num_out = static_cast<size_t *>(
        sampler_device->AllocWorkspace(sampler_ctx, sizeof(size_t)));
    size_t num_samples;

    LOG(DEBUG) << "DoGpuSample: size of out_src " << num_input * fanout;
    LOG(DEBUG) << "DoGpuSample: cuda out_src malloc "
               << ToReadableSize(num_input * fanout * sizeof(IdType));
    LOG(DEBUG) << "DoGpuSample: cuda out_dst malloc "
               << ToReadableSize(num_input * fanout * sizeof(IdType));
    LOG(DEBUG) << "DoGpuSample: cuda num_out malloc "
               << ToReadableSize(sizeof(size_t));

    // Sample a compact coo graph
    GpuSample(indptr, indices, input, num_input, fanout, out_src, out_dst,
              num_out, sampler_ctx, sample_stream, task->key);
    // Get nnz
    sampler_device->CopyDataFromTo(num_out, 0, &num_samples, 0, sizeof(size_t),
                                   sampler_ctx, CPU(), sample_stream);
    sampler_device->StreamSync(sampler_ctx, sample_stream);

    LOG(DEBUG) << "DoGpuSample: "
               << "layer " << i << " number of samples " << num_samples;

    double ns_time = t0.Passed();

    Timer t1;
    Timer t2;

    // Populate the hash table with newly sampled nodes
    IdType *unique = static_cast<IdType *>(sampler_device->AllocWorkspace(
        sampler_ctx, (num_samples + hash_table->NumItems()) * sizeof(IdType)));
    size_t num_unique;

    LOG(DEBUG) << "GpuSample: cuda unique malloc "
               << ToReadableSize((num_samples + +hash_table->NumItems()) *
                                 sizeof(IdType));

    hash_table->FillWithDuplicates(out_dst, num_samples, unique, &num_unique,
                                   sample_stream);

    double populate_time = t2.Passed();

    Timer t3;

    // Mapping edges
    IdType *new_src = static_cast<IdType *>(sampler_device->AllocWorkspace(
        sampler_ctx, num_samples * sizeof(IdType)));
    IdType *new_dst = static_cast<IdType *>(sampler_device->AllocWorkspace(
        sampler_ctx, num_samples * sizeof(IdType)));

    LOG(DEBUG) << "GpuSample: size of new_src " << num_samples;
    LOG(DEBUG) << "GpuSample: cuda new_src malloc "
               << ToReadableSize(num_samples * sizeof(IdType));
    LOG(DEBUG) << "GpuSample: cuda new_dst malloc "
               << ToReadableSize(num_samples * sizeof(IdType));

    MapEdges(out_src, new_src, out_dst, new_dst, num_samples,
             hash_table->DeviceHandle(), sampler_ctx, sample_stream);

    double map_edges_time = t3.Passed();
    double remap_time = t1.Passed();

    auto train_graph = std::make_shared<TrainGraph>();
    train_graph->num_row = num_unique;
    train_graph->num_column = num_input;
    train_graph->num_edge = num_samples;
    train_graph->col = Tensor::FromBlob(
        new_src, DataType::kI32, {num_samples}, sampler_ctx,
        "train_graph.row_cuda_sample_" + std::to_string(task->key) + "_" +
            std::to_string(i));
    train_graph->row = Tensor::FromBlob(
        new_dst, DataType::kI32, {num_samples}, sampler_ctx,
        "train_graph.dst_cuda_sample_" + std::to_string(task->key) + "_" +
            std::to_string(i));

    task->graphs[i] = train_graph;

    // Do some clean jobs
    sampler_device->FreeWorkspace(sampler_ctx, out_src);
    sampler_device->FreeWorkspace(sampler_ctx, out_dst);
    sampler_device->FreeWorkspace(sampler_ctx, num_out);

    LOG(DEBUG) << "layer " << i << " ns " << ns_time << " remap " << remap_time;

    Profiler::Get()->num_samples[task->key] += num_samples;
    Profiler::Get()->ns_time[task->key] += ns_time;
    Profiler::Get()->remap_time[task->key] += remap_time;
    Profiler::Get()->populate_time[task->key] += populate_time;
    Profiler::Get()->map_node_time[task->key] += 0;
    Profiler::Get()->map_edge_time[task->key] += map_edges_time;

    cur_input = Tensor::FromBlob(
        (void *)unique, DataType::kI32, {num_unique}, sampler_ctx,
        "cur_input_unique_cuda_" + std::to_string(task->key) + "_" +
            std::to_string(i));
    LOG(DEBUG) << "GpuSample: finish layer " << i;
  }

  task->input_nodes = cur_input;

  LOG(DEBUG) << "SampleLoop: process task with key " << task->key;
}

void DoGraphCopy(TaskPtr task) {
  auto sampler_ctx = GpuEngine::Get()->GetSamplerCtx();
  auto trainer_ctx = GpuEngine::Get()->GetTrainerCtx();
  auto sampler_device = Device::Get(sampler_ctx);
  auto copy_stream = GpuEngine::Get()->GetCopyStream();

  for (size_t i = 0; i < task->graphs.size(); i++) {
    auto graph = task->graphs[i];
    auto train_row =
        Tensor::Empty(graph->row->Type(), graph->row->Shape(), trainer_ctx,
                      "train_graph.row_cuda_train_" +
                          std::to_string(task->key) + "_" + std::to_string(i));
    auto train_col =
        Tensor::Empty(graph->col->Type(), graph->col->Shape(), trainer_ctx,
                      "train_graph.col_cuda_train_" +
                          std::to_string(task->key) + "_" + std::to_string(i));

    LOG(DEBUG) << "GraphCopyDevice2DeviceLoop: cuda train_row malloc "
               << ToReadableSize(graph->row->NumBytes());
    LOG(DEBUG) << "GraphCopyDevice2DeviceLoop: cuda train_col malloc "
               << ToReadableSize(graph->col->NumBytes());

    sampler_device->CopyDataFromTo(graph->row->Data(), 0,
                                   train_row->MutableData(), 0,
                                   graph->row->NumBytes(), graph->row->Ctx(),
                                   train_row->Ctx(), copy_stream);
    sampler_device->CopyDataFromTo(graph->col->Data(), 0,
                                   train_col->MutableData(), 0,
                                   graph->col->NumBytes(), graph->col->Ctx(),
                                   train_col->Ctx(), copy_stream);
    sampler_device->StreamSync(trainer_ctx, copy_stream);

    graph->row = train_row;
    graph->col = train_col;
  }

  LOG(DEBUG) << "GraphCopyDevice2Device: process task with key " << task->key;
}

void DoIdCopy(TaskPtr task) {
  auto sampler_ctx = GpuEngine::Get()->GetSamplerCtx();
  auto sampler_device = Device::Get(sampler_ctx);
  auto copy_stream = GpuEngine::Get()->GetCopyStream();

  auto input_nodes =
      Tensor::Empty(task->input_nodes->Type(), task->input_nodes->Shape(),
                    CPU(), "task.input_nodes_cpu_" + std::to_string(task->key));
  auto output_nodes = Tensor::Empty(
      task->output_nodes->Type(), task->output_nodes->Shape(), CPU(),
      "task.output_nodes_cpu_" + std::to_string(task->key));
  LOG(DEBUG) << "IdCopyDevice2Host input_nodes cpu malloc "
             << ToReadableSize(input_nodes->NumBytes());
  LOG(DEBUG) << "IdCopyDevice2Host output_nodes cpu malloc "
             << ToReadableSize(output_nodes->NumBytes());

  sampler_device->CopyDataFromTo(
      task->input_nodes->Data(), 0, input_nodes->MutableData(), 0,
      task->input_nodes->NumBytes(), task->input_nodes->Ctx(),
      input_nodes->Ctx(), copy_stream);
  sampler_device->CopyDataFromTo(
      task->output_nodes->Data(), 0, output_nodes->MutableData(), 0,
      task->output_nodes->NumBytes(), task->output_nodes->Ctx(),
      output_nodes->Ctx(), copy_stream);

  sampler_device->StreamSync(sampler_ctx, copy_stream);

  LOG(DEBUG) << "IdCopyDevice2Host: process task with key " << task->key;
}

void DoFeatureExtract(TaskPtr task) {
  auto dataset = GpuEngine::Get()->GetGraphDataset();
  auto input_nodes = task->input_nodes;
  auto output_nodes = task->output_nodes;

  auto feat = dataset->feat;
  auto label = dataset->label;

  auto feat_dim = dataset->feat->Shape()[1];
  auto feat_type = dataset->feat->Type();
  auto label_type = dataset->label->Type();

  auto input_data = reinterpret_cast<const IdType *>(input_nodes->Data());
  auto output_data = reinterpret_cast<const IdType *>(output_nodes->Data());
  auto num_input = input_nodes->Shape()[0];
  auto num_ouput = output_nodes->Shape()[0];

  task->input_feat =
      Tensor::Empty(feat_type, {num_input, feat_dim}, CPU(),
                    "task.input_feat_cpu_" + std::to_string(task->key));
  task->output_label =
      Tensor::Empty(label_type, {num_ouput}, CPU(),
                    "task.output_label_cpu" + std::to_string(task->key));

  auto extractor = GpuEngine::Get()->GetExtractor();

  auto feat_dst = task->input_feat->MutableData();
  auto feat_src = dataset->feat->Data();
  extractor->extract(feat_dst, feat_src, input_data, num_input, feat_dim,
                     feat_type);

  auto label_dst = task->output_label->MutableData();
  auto label_src = dataset->label->Data();
  extractor->extract(label_dst, label_src, output_data, num_ouput, 1,
                     label_type);

  LOG(DEBUG) << "HostFeatureExtract: process task with key " << task->key;
}

void DoFeatureCopy(TaskPtr task) {
  auto sampler_ctx = GpuEngine::Get()->GetSamplerCtx();
  auto sampler_device = Device::Get(sampler_ctx);
  auto trainer_ctx = GpuEngine::Get()->GetTrainerCtx();
  auto copy_stream = GpuEngine::Get()->GetCopyStream();

  auto cpu_feat = task->input_feat;
  auto cpu_label = task->output_label;

  auto train_feat =
      Tensor::Empty(cpu_feat->Type(), cpu_feat->Shape(), trainer_ctx,
                    "task.train_feat_cuda_" + std::to_string(task->key));
  auto train_label =
      Tensor::Empty(cpu_label->Type(), cpu_label->Shape(), trainer_ctx,
                    "task.train_label_cuda" + std::to_string(task->key));
  sampler_device->CopyDataFromTo(cpu_feat->Data(), 0, train_feat->MutableData(),
                                 0, cpu_feat->NumBytes(), cpu_feat->Ctx(),
                                 train_feat->Ctx(), copy_stream);
  sampler_device->CopyDataFromTo(
      cpu_label->Data(), 0, train_label->MutableData(), 0,
      cpu_label->NumBytes(), cpu_label->Ctx(), train_label->Ctx(), copy_stream);
  sampler_device->StreamSync(sampler_ctx, copy_stream);

  task->input_feat = train_feat;
  task->output_label = train_label;

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
