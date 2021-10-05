#include "cpu_loops.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <numeric>

#include "../cuda/cuda_function.h"
#include "../device.h"
#include "../logging.h"
#include "../profiler.h"
#include "../run_config.h"
#include "../timer.h"
#include "cpu_engine.h"
#include "cpu_function.h"
#include "cpu_hashtable.h"

namespace samgraph {
namespace common {
namespace cpu {

TaskPtr DoShuffle() {
  auto s = CPUEngine::Get()->GetShuffler();
  auto batch = s->GetBatch();

  if (batch) {
    auto task = std::make_shared<Task>();
    task->key = CPUEngine::Get()->GetBatchKey(s->Epoch(), s->Step());
    task->output_nodes = batch;
    return task;
  } else {
    return nullptr;
  }
}

void DoCPUSample(TaskPtr task) {
  auto fanouts = CPUEngine::Get()->GetFanout();
  auto num_layers = fanouts.size();
  auto last_layer_idx = num_layers - 1;

  auto dataset = CPUEngine::Get()->GetGraphDataset();
  auto cpu_device = Device::Get(CPU());

  auto hash_table = CPUEngine::Get()->GetHashTable();
  hash_table->Reset();

  size_t num_train_node = task->output_nodes->Shape()[0];
  hash_table->Populate(static_cast<const IdType *>(task->output_nodes->Data()),
                       num_train_node);

  task->graphs.resize(num_layers);

  const IdType *indptr = static_cast<const IdType *>(dataset->indptr->Data());
  const IdType *indices = static_cast<const IdType *>(dataset->indices->Data());
  IdType *mutable_indices =
      static_cast<IdType *>(dataset->indices->MutableData());

  auto cur_input = task->output_nodes;
  size_t last_layer_num_unique = 0;
  size_t total_num_samples = 0;
  for (int i = last_layer_idx; i >= 0; i--) {
    Timer t0;
    const size_t fanout = fanouts[i];
    const IdType *input = static_cast<const IdType *>(cur_input->Data());
    const size_t num_input = cur_input->Shape()[0];
    LOG(DEBUG) << "CPUSample: begin sample layer " << i;

    IdType *out_src = static_cast<IdType *>(
        cpu_device->AllocWorkspace(CPU(), num_input * fanout * sizeof(IdType)));
    IdType *out_dst = static_cast<IdType *>(
        cpu_device->AllocWorkspace(CPU(), num_input * fanout * sizeof(IdType)));
    size_t num_out;
    LOG(DEBUG) << "CPUSample: cpu out_src malloc "
               << ToReadableSize(num_input * fanout * sizeof(IdType));
    LOG(DEBUG) << "CPUSample: cpu out_src malloc "
               << ToReadableSize(num_input * fanout * sizeof(IdType));

    // Sample a compact coo graph
    switch (RunConfig::sample_type) {
      case kKHop0:
        CPUSampleKHop0(indptr, indices, input, num_input, out_src, out_dst,
                       &num_out, fanout);
        break;
      case kKHop2:
        CPUSampleKHop2(indptr, mutable_indices, input, num_input, out_src,
                       out_dst, &num_out, fanout);
        break;
      default:
        CHECK(0);
    }

    LOG(DEBUG) << "CPUSample: num_out " << num_out;
    double core_sample_time = t0.Passed();

    Timer t1;
    Timer t2;

    // Populate the hash table with newly sampled nodes
    hash_table->Populate(out_dst, num_out);

    double populate_time = t2.Passed();

    Timer t3;
    size_t num_unique = hash_table->NumItems();
    LOG(DEBUG) << "CPUSample: num_unique " << num_unique;
    IdType *unique = static_cast<IdType *>(
        cpu_device->AllocWorkspace(CPU(), num_unique * sizeof(IdType)));
    hash_table->MapNodes(unique, num_unique);

    double map_nodes_time = t3.Passed();

    Timer t4;
    if (i == (int)last_layer_idx) {
      last_layer_num_unique = num_unique;
    }
    // Mapping edges
    IdType *new_src = static_cast<IdType *>(
        cpu_device->AllocWorkspace(CPU(), num_out * sizeof(IdType)));
    IdType *new_dst = static_cast<IdType *>(
        cpu_device->AllocWorkspace(CPU(), num_out * sizeof(IdType)));
    LOG(DEBUG) << "CPUSample: cpu new_src malloc "
               << ToReadableSize(num_out * sizeof(IdType));
    LOG(DEBUG) << "CPUSample: cpu new_src malloc "
               << ToReadableSize(num_out * sizeof(IdType));
    hash_table->MapEdges(out_src, out_dst, num_out, new_src, new_dst);

    double map_edges_time = t4.Passed();

    double remap_time = t1.Passed();

    auto train_graph = std::make_shared<TrainGraph>();
    train_graph->row = Tensor::FromBlob(
        new_dst, DataType::kI32, {num_out}, CPU(),
        "train_graph.row_cpu_sample_" + std::to_string(task->key) + "_" +
            std::to_string(i));
    train_graph->col = Tensor::FromBlob(
        new_src, DataType::kI32, {num_out}, CPU(),
        "train_graph.col_cpu_sample_" + std::to_string(task->key) + "_" +
            std::to_string(i));
    train_graph->num_src = num_unique;
    train_graph->num_dst = num_input;
    train_graph->num_edge = num_out;

    task->graphs[i] = train_graph;
    cur_input =
        Tensor::FromBlob((void *)unique, DataType::kI32, {num_unique}, CPU(),
                         "cur_input_unique_cpu_" + std::to_string(task->key) +
                             "_" + std::to_string(i));
    total_num_samples += num_out;
    cpu_device->FreeWorkspace(CPU(), out_src);
    cpu_device->FreeWorkspace(CPU(), out_dst);

    Profiler::Get().LogStepAdd(task->key, kLogL2CoreSampleTime,
                               core_sample_time);
    Profiler::Get().LogStepAdd(task->key, kLogL2IdRemapTime, remap_time);
    Profiler::Get().LogStepAdd(task->key, kLogL3RemapPopulateTime,
                               populate_time);
    Profiler::Get().LogStepAdd(task->key, kLogL3RemapMapNodeTime,
                               map_nodes_time);
    Profiler::Get().LogStepAdd(task->key, kLogL3RemapMapEdgeTime,
                               map_edges_time);

    LOG(DEBUG) << "CPUSample: finish layer " << i;
  }

  task->input_nodes = cur_input;
  Profiler::Get().LogStep(task->key, kLogL1NumNode,
                          static_cast<double>(task->input_nodes->Shape()[0]));
  Profiler::Get().LogStep(task->key, kLogL1NumSample, total_num_samples);
  Profiler::Get().LogStep(task->key, kLogL2LastLayerSize,
                          static_cast<double>(last_layer_num_unique));
}

void DoFeatureExtract(TaskPtr task) {
  auto dataset = CPUEngine::Get()->GetGraphDataset();

  auto input_nodes = task->input_nodes;
  auto output_nodes = task->output_nodes;

  auto feat_dim = dataset->feat->Shape()[1];
  auto feat_type = dataset->feat->Type();
  auto label_type = dataset->label->Type();

  auto input_data = reinterpret_cast<const IdType *>(input_nodes->Data());
  auto output_data = reinterpret_cast<const IdType *>(output_nodes->Data());
  auto num_input = input_nodes->Shape()[0];
  auto num_ouput = output_nodes->Shape()[0];

  auto feat = Tensor::Empty(feat_type, {num_input, feat_dim}, CPU(),
                            "task.input_feat_cpu_" + std::to_string(task->key));
  auto label =
      Tensor::Empty(label_type, {num_ouput}, CPU(),
                    "task.output_label_cpu" + std::to_string(task->key));

  auto feat_dst = feat->MutableData();
  auto feat_src = dataset->feat->Data();
  if (RunConfig::option_empty_feat != 0) {
    CPUMockExtract(feat_dst, feat_src, input_data, num_input, feat_dim, feat_type);
  } else {
    CPUExtract(feat_dst, feat_src, input_data, num_input, feat_dim, feat_type);
  }
  auto label_dst = label->MutableData();
  auto label_src = dataset->label->Data();
  CPUExtract(label_dst, label_src, output_data, num_ouput, 1, label_type);

  task->input_feat = feat;
  task->output_label = label;
}

void DoGraphCopy(TaskPtr task) {
  auto trainer_ctx = CPUEngine::Get()->GetTrainerCtx();
  auto trainer_device = Device::Get(trainer_ctx);
  auto work_stream = CPUEngine::Get()->GetWorkStream();

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

    trainer_device->CopyDataFromTo(
        graph->row->Data(), 0, train_row->MutableData(), 0,
        graph->row->NumBytes(), CPU(), trainer_ctx, work_stream);
    trainer_device->CopyDataFromTo(
        graph->col->Data(), 0, train_col->MutableData(), 0,
        graph->col->NumBytes(), CPU(), trainer_ctx, work_stream);
    trainer_device->StreamSync(trainer_ctx, work_stream);

    graph->row = train_row;
    graph->col = train_col;

    Profiler::Get().LogStepAdd(task->key, kLogL1GraphBytes,
                               train_row->NumBytes() + train_col->NumBytes());
  }
}

void DoFeatureCopy(TaskPtr task) {
  auto trainer_ctx = CPUEngine::Get()->GetTrainerCtx();
  auto trainer_device = Device::Get(trainer_ctx);
  auto work_stream = CPUEngine::Get()->GetWorkStream();

  auto feat = task->input_feat;
  auto label = task->output_label;

  auto train_feat =
      Tensor::Empty(feat->Type(), feat->Shape(), trainer_ctx,
                    "task.train_feat_cuda_" + std::to_string(task->key));
  auto train_label =
      Tensor::Empty(label->Type(), label->Shape(), trainer_ctx,
                    "task.train_label_cuda" + std::to_string(task->key));

  trainer_device->CopyDataFromTo(feat->Data(), 0, train_feat->MutableData(), 0,
                                 feat->NumBytes(), CPU(), trainer_ctx,
                                 work_stream);

  trainer_device->CopyDataFromTo(label->Data(), 0, train_label->MutableData(),
                                 0, label->NumBytes(), CPU(), trainer_ctx,
                                 work_stream);
  trainer_device->StreamSync(trainer_ctx, work_stream);

  task->input_feat = train_feat;
  task->output_label = train_label;

  Profiler::Get().LogStep(task->key, kLogL1FeatureBytes,
                          train_feat->NumBytes());
  Profiler::Get().LogStep(task->key, kLogL1LabelBytes, train_label->NumBytes());
  Profiler::Get().LogEpochAdd(task->key, kLogEpochFeatureBytes,
                              train_feat->NumBytes());
  Profiler::Get().LogEpochAdd(task->key, kLogEpochMissBytes,train_feat->NumBytes());
}

void DoCacheIdCopy(TaskPtr task) {
  auto trainer_ctx = CPUEngine::Get()->GetTrainerCtx();
  auto trainer_device = Device::Get(trainer_ctx);
  auto stream = CPUEngine::Get()->GetWorkStream();

  auto input_nodes = Tensor::Empty(
      task->input_nodes->Type(), task->input_nodes->Shape(), trainer_ctx,
      "task.output_nodes_cuda_" + std::to_string(task->key));
  LOG(DEBUG) << "DoCacheIdCopy input_nodes cuda malloc "
             << ToReadableSize(input_nodes->NumBytes());
  auto output_nodes = Tensor::Empty(
      task->output_nodes->Type(), task->output_nodes->Shape(), trainer_ctx,
      "task.output_nodes_cuda_" + std::to_string(task->key));
  LOG(DEBUG) << "DoCacheIdCopy output_nodes cuda malloc "
             << ToReadableSize(output_nodes->NumBytes());

  trainer_device->CopyDataFromTo(
      task->input_nodes->Data(), 0, input_nodes->MutableData(), 0,
      task->input_nodes->NumBytes(), task->input_nodes->Ctx(),
      input_nodes->Ctx(), stream);
  trainer_device->CopyDataFromTo(
      task->output_nodes->Data(), 0, output_nodes->MutableData(), 0,
      task->output_nodes->NumBytes(), task->output_nodes->Ctx(),
      output_nodes->Ctx(), stream);

  trainer_device->StreamSync(trainer_ctx, stream);

  task->input_nodes = input_nodes;
  task->output_nodes = output_nodes;

  Profiler::Get().LogStepAdd(
      task->key, kLogL1IdBytes,
      input_nodes->NumBytes() + output_nodes->NumBytes());

  LOG(DEBUG) << "DoCacheIdCopy: process task with key " << task->key;
}

void DoGPULabelExtract(TaskPtr task) {
  auto dataset = CPUEngine::Get()->GetGraphDataset();

  auto trainer_ctx = CPUEngine::Get()->GetTrainerCtx();
  auto trainer_device = Device::Get(trainer_ctx);
  auto stream = CPUEngine::Get()->GetWorkStream();

  auto output_nodes = task->output_nodes;
  auto label = dataset->label;
  auto label_type = dataset->label->Type();

  auto output_data = reinterpret_cast<const IdType *>(output_nodes->Data());
  auto num_ouput = output_nodes->Shape()[0];

  auto train_label =
      Tensor::Empty(label_type, {num_ouput}, trainer_ctx,
                    "task.train_label_cuda" + std::to_string(task->key));

  void *label_dst = train_label->MutableData();
  const void *label_src = dataset->label->Data();

  CHECK_EQ(output_nodes->Ctx().device_type, trainer_ctx.device_type);
  CHECK_EQ(output_nodes->Ctx().device_id, trainer_ctx.device_id);
  CHECK_EQ(dataset->label->Ctx().device_type, trainer_ctx.device_type);
  CHECK_EQ(dataset->label->Ctx().device_id, trainer_ctx.device_id);

  cuda::GPUExtract(label_dst, label_src, output_data, num_ouput, 1, label_type,
                   trainer_ctx, stream, task->key);

  LOG(DEBUG) << "DoGPULabelExtract: process task with key " << task->key;
  trainer_device->StreamSync(trainer_ctx, stream);

  task->output_label = train_label;

  Profiler::Get().LogStep(task->key, kLogL1LabelBytes, train_label->NumBytes());
}

void DoCacheFeatureExtractCopy(TaskPtr task) {
  auto trainer_ctx = CPUEngine::Get()->GetTrainerCtx();
  auto trainer_device = Device::Get(trainer_ctx);
  auto cpu_device = Device::Get(CPU());
  auto stream = CPUEngine::Get()->GetWorkStream();

  auto dataset = CPUEngine::Get()->GetGraphDataset();
  auto cache_manager = CPUEngine::Get()->GetCacheManager();

  auto input_nodes = task->input_nodes;
  auto feat = dataset->feat;
  auto feat_dim = dataset->feat->Shape()[1];
  auto feat_type = dataset->feat->Type();

  auto input_data = reinterpret_cast<const IdType *>(input_nodes->Data());
  auto num_input = input_nodes->Shape()[0];

  CHECK_EQ(input_nodes->Ctx(), trainer_ctx);

  auto train_feat =
      Tensor::Empty(feat_type, {num_input, feat_dim}, trainer_ctx,
                    "task.train_feat_cuda_" + std::to_string(task->key));

  // 0. Get index of miss data and cache data
  // feature data has cache, so we only need to extract the miss data
  Timer t0;

  IdType *trainer_output_miss_src_index = static_cast<IdType *>(
      trainer_device->AllocWorkspace(trainer_ctx, sizeof(IdType) * num_input));
  IdType *trainer_output_miss_dst_index = static_cast<IdType *>(
      trainer_device->AllocWorkspace(trainer_ctx, sizeof(IdType) * num_input));
  IdType *trainer_output_cache_src_index = static_cast<IdType *>(
      trainer_device->AllocWorkspace(trainer_ctx, sizeof(IdType) * num_input));
  IdType *trainer_output_cache_dst_index = static_cast<IdType *>(
      trainer_device->AllocWorkspace(trainer_ctx, sizeof(IdType) * num_input));

  size_t num_output_miss;
  size_t num_output_cache;

  cache_manager->GetMissCacheIndex(
      trainer_output_miss_src_index, trainer_output_miss_dst_index,
      &num_output_miss, trainer_output_cache_src_index,
      trainer_output_cache_dst_index, &num_output_cache, input_data, num_input,
      stream);

  CHECK_EQ(num_output_miss + num_output_cache, num_input);

  double get_index_time = t0.Passed();

  // 1. Move the miss index
  Timer t1;

  IdType *cpu_output_miss_src_index = static_cast<IdType *>(
      cpu_device->AllocWorkspace(CPU(), sizeof(IdType) * num_output_miss));

  trainer_device->CopyDataFromTo(
      trainer_output_miss_src_index, 0, cpu_output_miss_src_index, 0,
      num_output_miss * sizeof(IdType), trainer_ctx, CPU(), stream);

  trainer_device->StreamSync(trainer_ctx, stream);
  trainer_device->FreeWorkspace(trainer_ctx, trainer_output_miss_src_index);

  double copy_idx_time = t1.Passed();

  // 2. Extract the miss data
  Timer t2;

  void *cpu_output_miss = cpu_device->AllocWorkspace(
      CPU(), GetTensorBytes(feat_type, {num_output_miss, feat_dim}));
  void *trainer_output_miss = trainer_device->AllocWorkspace(
      trainer_ctx, GetTensorBytes(feat_type, {num_output_miss, feat_dim}));

  cache_manager->ExtractMissData(cpu_output_miss, cpu_output_miss_src_index,
                                 num_output_miss);

  double extract_miss_time = t2.Passed();

  // 3. Copy the miss data
  Timer t3;

  trainer_device->CopyDataFromTo(
      cpu_output_miss, 0, trainer_output_miss, 0,
      GetTensorBytes(feat_type, {num_output_miss, feat_dim}), CPU(),
      trainer_ctx, stream);
  trainer_device->StreamSync(trainer_ctx, stream);

  cpu_device->FreeWorkspace(CPU(), cpu_output_miss);

  double copy_miss_time = t3.Passed();

  // 4. Combine miss data
  Timer t4;
  cache_manager->CombineMissData(train_feat->MutableData(), trainer_output_miss,
                                 trainer_output_miss_dst_index, num_output_miss,
                                 stream);
  trainer_device->StreamSync(trainer_ctx, stream);

  double combine_miss_time = t4.Passed();

  LOG(DEBUG) << "num_cache " << num_output_cache;

  // 5. Combine cache data
  Timer t5;
  cache_manager->CombineCacheData(
      train_feat->MutableData(), trainer_output_cache_src_index,
      trainer_output_cache_dst_index, num_output_cache, stream);
  trainer_device->StreamSync(trainer_ctx, stream);

  double combine_cache_time = t5.Passed();

  task->input_feat = train_feat;

  // 5. Free space
  cpu_device->FreeWorkspace(CPU(), cpu_output_miss_src_index);
  trainer_device->FreeWorkspace(trainer_ctx, trainer_output_miss);
  trainer_device->FreeWorkspace(trainer_ctx, trainer_output_miss_dst_index);
  trainer_device->FreeWorkspace(trainer_ctx, trainer_output_cache_src_index);
  trainer_device->FreeWorkspace(trainer_ctx, trainer_output_cache_dst_index);

  Profiler::Get().LogStep(task->key, kLogL1FeatureBytes,
                          train_feat->NumBytes());
  Profiler::Get().LogStep(
      task->key, kLogL1MissBytes,
      GetTensorBytes(feat_type, {num_output_miss, feat_dim}));
  Profiler::Get().LogStep(task->key, kLogL3CacheGetIndexTime,
                          get_index_time);  // t0, step 0
  Profiler::Get().LogStep(task->key, KLogL3CacheCopyIndexTime,
                          copy_idx_time);  // t1, step 1
  Profiler::Get().LogStep(task->key, kLogL3CacheExtractMissTime,
                          extract_miss_time);  // t2, step2
  Profiler::Get().LogStep(task->key, kLogL3CacheCopyMissTime,
                          copy_miss_time);  // t3, step 3
  Profiler::Get().LogStep(task->key, kLogL3CacheCombineMissTime,
                          combine_miss_time);  // t4, step 4
  Profiler::Get().LogStep(task->key, kLogL3CacheCombineCacheTime,
                          combine_cache_time);  // t5, step 5
  Profiler::Get().LogEpochAdd(task->key, kLogEpochFeatureBytes,
                              train_feat->NumBytes());
  Profiler::Get().LogEpochAdd(
      task->key, kLogEpochMissBytes,
      GetTensorBytes(feat_type, {num_output_miss, feat_dim}));

  LOG(DEBUG) << "DoCacheFeatureCopy: process task with key " << task->key;
}

}  // namespace cpu
}  // namespace common
}  // namespace samgraph
