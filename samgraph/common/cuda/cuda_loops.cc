#include "cuda_loops.h"

#include <cuda_runtime.h>

#include <chrono>
#include <numeric>
#include <thread>

#include "../device.h"
#include "../function.h"
#include "../logging.h"
#include "../profiler.h"
#include "../run_config.h"
#include "../timer.h"
#include "cuda_common.h"
#include "cuda_engine.h"
#include "cuda_function.h"
#include "cuda_hashtable.h"

namespace samgraph {
namespace common {
namespace cuda {

using TaskPtr = std::shared_ptr<Task>;

TaskPtr DoShuffle() {
  auto s = GPUEngine::Get()->GetShuffler();
  auto batch = s->GetBatch();

  if (batch) {
    auto task = std::make_shared<Task>();
    task->key = GPUEngine::Get()->GetBatchKey(s->Epoch(), s->Step());
    task->output_nodes = batch;
    LOG(DEBUG) << "DoShuffle: process task with key " << task->key;
    return task;
  } else {
    return nullptr;
  }
}

void DoGPUSample(TaskPtr task) {
  auto fanouts = GPUEngine::Get()->GetFanout();
  auto num_layers = fanouts.size();
  auto last_layer_idx = num_layers - 1;

  auto dataset = GPUEngine::Get()->GetGraphDataset();
  auto sampler_ctx = GPUEngine::Get()->GetSamplerCtx();
  auto sampler_device = Device::Get(sampler_ctx);
  auto sample_stream = GPUEngine::Get()->GetSampleStream();

  OrderedHashTable *hash_table = GPUEngine::Get()->GetHashtable();
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
    LOG(DEBUG) << "DoGPUSample: begin sample layer " << i;

    IdType *out_src = static_cast<IdType *>(sampler_device->AllocWorkspace(
        sampler_ctx, num_input * fanout * sizeof(IdType)));
    IdType *out_dst = static_cast<IdType *>(sampler_device->AllocWorkspace(
        sampler_ctx, num_input * fanout * sizeof(IdType)));
    size_t *num_out = static_cast<size_t *>(
        sampler_device->AllocWorkspace(sampler_ctx, sizeof(size_t)));
    size_t num_samples;

    LOG(DEBUG) << "DoGPUSample: size of out_src " << num_input * fanout;
    LOG(DEBUG) << "DoGPUSample: cuda out_src malloc "
               << ToReadableSize(num_input * fanout * sizeof(IdType));
    LOG(DEBUG) << "DoGPUSample: cuda out_dst malloc "
               << ToReadableSize(num_input * fanout * sizeof(IdType));
    LOG(DEBUG) << "DoGPUSample: cuda num_out malloc "
               << ToReadableSize(sizeof(size_t));

    // Sample a compact coo graph
    if (dataset->weighted_edge) {
      GPUWeightedSample(indptr, indices, input, num_input, fanout, out_src,
                        out_dst, num_out, sampler_ctx, sample_stream,
                        task->key);
    } else {
      GPUSample(indptr, indices, input, num_input, fanout, out_src, out_dst,
                num_out, sampler_ctx, sample_stream, task->key);
    }

    // Get nnz
    sampler_device->CopyDataFromTo(num_out, 0, &num_samples, 0, sizeof(size_t),
                                   sampler_ctx, CPU(), sample_stream);
    sampler_device->StreamSync(sampler_ctx, sample_stream);

    LOG(DEBUG) << "DoGPUSample: "
               << "layer " << i << " number of samples " << num_samples;

    double core_sample_time = t0.Passed();

    Timer t1;
    Timer t2;

    // Populate the hash table with newly sampled nodes
    IdType *unique = static_cast<IdType *>(sampler_device->AllocWorkspace(
        sampler_ctx, (num_samples + hash_table->NumItems()) * sizeof(IdType)));
    size_t num_unique;

    LOG(DEBUG) << "GPUSample: cuda unique malloc "
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

    LOG(DEBUG) << "GPUSample: size of new_src " << num_samples;
    LOG(DEBUG) << "GPUSample: cuda new_src malloc "
               << ToReadableSize(num_samples * sizeof(IdType));
    LOG(DEBUG) << "GPUSample: cuda new_dst malloc "
               << ToReadableSize(num_samples * sizeof(IdType));

    GPUMapEdges(out_src, new_src, out_dst, new_dst, num_samples,
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

    Profiler::Get().LogAdd(task->key, kLogL2CoreSampleTime, core_sample_time);
    Profiler::Get().LogAdd(task->key, kLogL2IdRemapTime, remap_time);
    Profiler::Get().LogAdd(task->key, kLogL3RemapPopulateTime, populate_time);
    Profiler::Get().LogAdd(task->key, kLogL3RemapMapNodeTime, 0);
    Profiler::Get().LogAdd(task->key, kLogL3RemapMapEdgeTime, map_edges_time);

    cur_input = Tensor::FromBlob(
        (void *)unique, DataType::kI32, {num_unique}, sampler_ctx,
        "cur_input_unique_cuda_" + std::to_string(task->key) + "_" +
            std::to_string(i));
    LOG(DEBUG) << "GPUSample: finish layer " << i;
  }

  task->input_nodes = cur_input;
  Profiler::Get().Log(task->key, kLogL1NumNode,
                      static_cast<double>(task->input_nodes->Shape()[0]));

  LOG(DEBUG) << "SampleLoop: process task with key " << task->key;
}

void DoGraphCopy(TaskPtr task) {
  auto sampler_ctx = GPUEngine::Get()->GetSamplerCtx();
  auto trainer_ctx = GPUEngine::Get()->GetTrainerCtx();
  auto sampler_device = Device::Get(sampler_ctx);
  auto copy_stream = GPUEngine::Get()->GetSamplerCopyStream();

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

    Profiler::Get().LogAdd(task->key, kLogL1GraphBytes,
                           train_row->NumBytes() + train_col->NumBytes());
  }

  LOG(DEBUG) << "GraphCopyDevice2Device: process task with key " << task->key;
}

void DoIdCopy(TaskPtr task) {
  auto sampler_ctx = GPUEngine::Get()->GetSamplerCtx();
  auto sampler_device = Device::Get(sampler_ctx);
  auto copy_stream = GPUEngine::Get()->GetSamplerCopyStream();

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

  task->input_nodes = input_nodes;
  task->output_nodes = output_nodes;

  Profiler::Get().LogAdd(task->key, kLogL1IdBytes,
                         input_nodes->NumBytes() + output_nodes->NumBytes());

  sampler_device->StreamSync(sampler_ctx, copy_stream);

  LOG(DEBUG) << "IdCopyDevice2Host: process task with key " << task->key;
}

void DoFeatureExtract(TaskPtr task) {
  auto dataset = GPUEngine::Get()->GetGraphDataset();

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

  auto feat_dst = task->input_feat->MutableData();
  auto feat_src = dataset->feat->Data();

  cpu::CPUExtract(feat_dst, feat_src, input_data, num_input, feat_dim,
                  feat_type);

  auto label_dst = task->output_label->MutableData();
  auto label_src = dataset->label->Data();
  cpu::CPUExtract(label_dst, label_src, output_data, num_ouput, 1, label_type);

  if (RunConfig::option_log_node_access) {
    Profiler::Get().LogNodeAccess(task->key, input_data, num_input);
  }

  LOG(DEBUG) << "HostFeatureExtract: process task with key " << task->key;
}

void DoFeatureCopy(TaskPtr task) {
  auto trainer_ctx = GPUEngine::Get()->GetTrainerCtx();
  auto trainer_device = Device::Get(trainer_ctx);
  auto copy_stream = GPUEngine::Get()->GetTrainerCopyStream();

  auto cpu_feat = task->input_feat;
  auto cpu_label = task->output_label;

  auto train_feat =
      Tensor::Empty(cpu_feat->Type(), cpu_feat->Shape(), trainer_ctx,
                    "task.train_feat_cuda_" + std::to_string(task->key));
  auto train_label =
      Tensor::Empty(cpu_label->Type(), cpu_label->Shape(), trainer_ctx,
                    "task.train_label_cuda" + std::to_string(task->key));
  trainer_device->CopyDataFromTo(cpu_feat->Data(), 0, train_feat->MutableData(),
                                 0, cpu_feat->NumBytes(), cpu_feat->Ctx(),
                                 train_feat->Ctx(), copy_stream);
  trainer_device->CopyDataFromTo(
      cpu_label->Data(), 0, train_label->MutableData(), 0,
      cpu_label->NumBytes(), cpu_label->Ctx(), train_label->Ctx(), copy_stream);
  trainer_device->StreamSync(trainer_ctx, copy_stream);

  task->input_feat = train_feat;
  task->output_label = train_label;

  Profiler::Get().Log(task->key, kLogL1FeatureBytes, train_feat->NumBytes());
  Profiler::Get().Log(task->key, kLogL1LabelBytes, train_label->NumBytes());

  LOG(DEBUG) << "FeatureCopyHost2Device: process task with key " << task->key;
}

void DoCacheFeatureCopy(TaskPtr task) {
  auto trainer_ctx = GPUEngine::Get()->GetTrainerCtx();
  auto trainer_device = Device::Get(trainer_ctx);
  auto cpu_device = Device::Get(CPU());
  auto copy_stream = GPUEngine::Get()->GetTrainerCopyStream();

  auto dataset = GPUEngine::Get()->GetGraphDataset();
  auto cache_manager = GPUEngine::Get()->GetCacheManager();

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

  auto train_feat =
      Tensor::Empty(feat_type, {num_input, feat_dim}, trainer_ctx,
                    "task.train_feat_cuda_" + std::to_string(task->key));

  // 1. Extract the miss data
  // feature data has cache, so we only need to extract the miss data
  Timer t0;
  void *output_miss = cpu_device->AllocWorkspace(
      CPU(), GetTensorBytes(feat_type, {num_input, feat_dim}));
  IdType *output_miss_index = static_cast<IdType *>(
      cpu_device->AllocWorkspace(CPU(), sizeof(IdType) * num_input));
  IdType *output_cache_src_index = static_cast<IdType *>(
      cpu_device->AllocWorkspace(CPU(), sizeof(IdType) * num_input));
  IdType *output_cache_dst_index = static_cast<IdType *>(
      cpu_device->AllocWorkspace(CPU(), sizeof(IdType) * num_input));
  size_t num_output_miss;
  size_t num_output_cache;

  cache_manager->ExtractMissData(
      output_miss, output_miss_index, &num_output_miss, output_cache_src_index,
      output_cache_dst_index, &num_output_cache, input_data, num_input);

  double cache_extract_time = t0.Passed();

  // 2. Copy the miss data, miss index, and cache index into GPU memory
  Timer t1;

  void *gpu_miss = trainer_device->AllocWorkspace(
      trainer_ctx, GetTensorBytes(feat_type, {num_output_miss, feat_dim}));
  IdType *gpu_miss_index = static_cast<IdType *>(trainer_device->AllocWorkspace(
      trainer_ctx, num_output_miss * sizeof(IdType)));
  IdType *gpu_cache_src_index =
      static_cast<IdType *>(trainer_device->AllocWorkspace(
          trainer_ctx, num_output_cache * sizeof(IdType)));
  IdType *gpu_cache_dst_index =
      static_cast<IdType *>(trainer_device->AllocWorkspace(
          trainer_ctx, num_output_cache * sizeof(IdType)));

  trainer_device->CopyDataFromTo(
      output_miss, 0, gpu_miss, 0,
      GetTensorBytes(feat_type, {num_output_miss, feat_dim}), CPU(),
      trainer_ctx, copy_stream);

  trainer_device->CopyDataFromTo(output_miss_index, 0, gpu_miss_index, 0,
                                 num_output_miss * sizeof(IdType), CPU(),
                                 trainer_ctx, copy_stream);
  trainer_device->CopyDataFromTo(output_cache_src_index, 0, gpu_cache_src_index,
                                 0, num_output_cache * sizeof(IdType), CPU(),
                                 trainer_ctx, copy_stream);
  trainer_device->CopyDataFromTo(output_cache_dst_index, 0, gpu_cache_dst_index,
                                 0, num_output_cache * sizeof(IdType), CPU(),
                                 trainer_ctx, copy_stream);

  trainer_device->StreamSync(trainer_ctx, copy_stream);

  cpu_device->FreeWorkspace(CPU(), output_miss);
  cpu_device->FreeWorkspace(CPU(), output_miss_index);
  cpu_device->FreeWorkspace(CPU(), output_cache_src_index);
  cpu_device->FreeWorkspace(CPU(), output_cache_dst_index);

  double cache_copy_miss_time = t1.Passed();

  // 3. Combine miss data
  Timer t2;
  cache_manager->CombineMissData(train_feat->MutableData(), gpu_miss,
                                 gpu_miss_index, num_output_miss, copy_stream);
  trainer_device->StreamSync(trainer_ctx, copy_stream);

  double cache_combine_miss_time = t2.Passed();

  // 4. Combine cache data
  Timer t3;
  cache_manager->CombineCacheData(train_feat->MutableData(),
                                  gpu_cache_src_index, gpu_cache_dst_index,
                                  num_output_cache, copy_stream);
  trainer_device->StreamSync(trainer_ctx, copy_stream);

  double cache_combine_cache_time = t3.Passed();

  // 5. Free space
  trainer_device->FreeWorkspace(trainer_ctx, gpu_miss);
  trainer_device->FreeWorkspace(trainer_ctx, gpu_miss_index);
  trainer_device->FreeWorkspace(trainer_ctx, gpu_cache_src_index);
  trainer_device->FreeWorkspace(trainer_ctx, gpu_cache_dst_index);

  // label data has no cache
  void *label_dst = cpu_device->AllocWorkspace(
      CPU(), GetTensorBytes(label_type, {num_ouput}));
  const void *label_src = dataset->label->Data();

  cpu::CPUExtract(label_dst, label_src, output_data, num_ouput, 1, label_type);

  cpu_device->FreeWorkspace(CPU(), label_dst);

  LOG(DEBUG) << "HostFeatureExtract: process task with key " << task->key;

  auto train_label =
      Tensor::Empty(label_type, {num_ouput}, trainer_ctx,
                    "task.train_label_cuda" + std::to_string(task->key));
  trainer_device->CopyDataFromTo(label_dst, 0, train_label->MutableData(), 0,
                                 GetTensorBytes(label_type, {num_ouput}), CPU(),
                                 train_label->Ctx(), copy_stream);
  trainer_device->StreamSync(trainer_ctx, copy_stream);

  task->input_feat = train_feat;
  task->output_label = train_label;

  Profiler::Get().Log(task->key, kLogL1FeatureBytes, train_feat->NumBytes());
  Profiler::Get().Log(task->key, kLogL1LabelBytes, train_label->NumBytes());

  Profiler::Get().Log(task->key, kLogL3CacheExtractTime, cache_extract_time);
  Profiler::Get().Log(task->key, kLogL3CacheCopyMissTime, cache_copy_miss_time);
  Profiler::Get().Log(task->key, kLogL3CacheCombineMissTime,
                      cache_combine_miss_time);
  Profiler::Get().Log(task->key, kLogL3CacheCombineCacheTime,
                      cache_combine_cache_time);

  if (RunConfig::option_log_node_access) {
    Profiler::Get().LogNodeAccess(task->key, input_data, num_input);
  }

  LOG(DEBUG) << "DoCacheFeatureCopy: process task with key " << task->key;
}

bool RunGPUSampleLoopOnce() {
  auto next_op = kDataCopy;
  auto next_q = GPUEngine::Get()->GetTaskQueue(next_op);
  if (next_q->Full()) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    return true;
  }

  Timer t0;
  auto task = DoShuffle();
  if (task) {
    double shuffle_time = t0.Passed();

    Timer t1;
    DoGPUSample(task);
    double sample_time = t1.Passed();

    next_q->AddTask(task);

    Profiler::Get().Log(task->key, kLogL1SampleTime,
                        shuffle_time + sample_time);
    Profiler::Get().Log(task->key, kLogL2ShuffleTime, shuffle_time);
  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }

  return true;
}

bool RunDataCopyLoopOnce() {
  auto graph_pool = GPUEngine::Get()->GetGraphPool();
  if (graph_pool->Full()) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    return true;
  }

  auto this_op = kDataCopy;
  auto q = GPUEngine::Get()->GetTaskQueue(this_op);
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
    graph_pool->Submit(task->key, task);

    Profiler::Get().Log(
        task->key, kLogL1CopyTime,
        graph_copy_time + id_copy_time + extract_time + feat_copy_time);
    Profiler::Get().Log(task->key, kLogL2GraphCopyTime, graph_copy_time);
    Profiler::Get().Log(task->key, kLogL2IdCopyTime, id_copy_time);
    Profiler::Get().Log(task->key, kLogL2ExtractTime, extract_time);
    Profiler::Get().Log(task->key, kLogL2FeatCopyTime, feat_copy_time);

  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }

  return true;
}

bool RunCacheDataCopyLoopOnce() {
  auto graph_pool = GPUEngine::Get()->GetGraphPool();
  if (graph_pool->Full()) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    return true;
  }

  auto this_op = kDataCopy;
  auto q = GPUEngine::Get()->GetTaskQueue(this_op);
  auto task = q->GetTask();

  if (task) {
    Timer t0;
    DoGraphCopy(task);
    double graph_copy_time = t0.Passed();

    Timer t1;
    DoIdCopy(task);
    double id_copy_time = t1.Passed();

    Timer t2;
    DoCacheFeatureCopy(task);
    double cache_feat_copy_time = t2.Passed();

    LOG(DEBUG) << "Submit with cache: process task with key " << task->key;
    graph_pool->Submit(task->key, task);

    Profiler::Get().Log(task->key, kLogL1CopyTime,
                        graph_copy_time + id_copy_time + cache_feat_copy_time);
    Profiler::Get().Log(task->key, kLogL2GraphCopyTime, graph_copy_time);
    Profiler::Get().Log(task->key, kLogL2IdCopyTime, id_copy_time);
    Profiler::Get().Log(task->key, kLogL2CacheCopyTime, cache_feat_copy_time);
  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }
  return true;
}

void GPUSampleLoop() {
  while (RunGPUSampleLoopOnce() && !GPUEngine::Get()->ShouldShutdown()) {
  }
  GPUEngine::Get()->ReportThreadFinish();
}

void DataCopyLoop() {
  LoopOnceFunction func;
  if (!RunConfig::UseGPUCache()) {
    func = RunDataCopyLoopOnce;
  } else {
    func = RunCacheDataCopyLoopOnce;
  }

  while (func() && !GPUEngine::Get()->ShouldShutdown()) {
  }

  GPUEngine::Get()->ReportThreadFinish();
}

}  // namespace cuda
}  // namespace common
}  // namespace samgraph
