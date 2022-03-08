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

#include "dist_loops.h"

#include "../device.h"
#include "../function.h"
#include "../logging.h"
#include "../profiler.h"
#include "../run_config.h"
#include "../timer.h"

#include "dist_engine.h"
#include "../cuda/cuda_hashtable.h"
#include "../cuda/cuda_function.h"

namespace samgraph {
namespace common {
namespace dist {

TaskPtr DoShuffle() {
  auto s = DistEngine::Get()->GetShuffler();
  auto batch = s->GetBatch();

  if (batch) {
    auto task = std::make_shared<Task>();
    // global key
    task->key = DistEngine::Get()->GetBatchKey(s->Epoch(), s->Step());
    task->output_nodes = batch;
    LOG(DEBUG) << "DoShuffle: process task with key " << task->key;
    return task;
  } else {
    return nullptr;
  }
}

void DoGPUSample(TaskPtr task) {
  auto fanouts = DistEngine::Get()->GetFanout();
  auto num_layers = fanouts.size();
  auto last_layer_idx = num_layers - 1;

  auto dataset = DistEngine::Get()->GetGraphDataset();
  auto sampler_ctx = DistEngine::Get()->GetSamplerCtx();
  auto sampler_device = Device::Get(sampler_ctx);
  auto sample_stream = DistEngine::Get()->GetSampleStream();

  auto random_states = DistEngine::Get()->GetRandomStates();
  auto frequency_hashmap = DistEngine::Get()->GetFrequencyHashmap();

  cuda::OrderedHashTable *hash_table = DistEngine::Get()->GetHashtable();
  hash_table->Reset(sample_stream);

  Timer t;
  auto output_nodes = task->output_nodes;
  size_t num_train_node = output_nodes->Shape()[0];
  hash_table->FillWithUnique(
      static_cast<const IdType *const>(output_nodes->Data()), num_train_node,
      sample_stream);
  task->graphs.resize(num_layers);
  double fill_unique_time = t.Passed();

  const IdType *indptr = static_cast<const IdType *>(dataset->indptr->Data());
  const IdType *indices = static_cast<const IdType *>(dataset->indices->Data());
  const float *prob_table =
      static_cast<const float *>(dataset->prob_table->Data());
  const IdType *alias_table =
      static_cast<const IdType *>(dataset->alias_table->Data());
  const float *prob_prefix_table =
      static_cast<const float *>(dataset->prob_prefix_table->Data());

  auto cur_input = task->output_nodes;
  size_t total_num_samples = 0;

  for (int i = last_layer_idx; i >= 0; i--) {
    Timer tlayer;
    Timer t0;
    const size_t fanout = fanouts[i];
    const IdType *input = static_cast<const IdType *>(cur_input->Data());
    const size_t num_input = cur_input->Shape()[0];
    LOG(DEBUG) << "DoGPUSample: begin sample layer " << i;

    IdType *out_src = static_cast<IdType *>(sampler_device->AllocWorkspace(
        sampler_ctx, num_input * fanout * sizeof(IdType)));
    IdType *out_dst = static_cast<IdType *>(sampler_device->AllocWorkspace(
        sampler_ctx, num_input * fanout * sizeof(IdType)));
    IdType *out_data = nullptr;
    if (RunConfig::sample_type == kRandomWalk) {
      out_data = static_cast<IdType *>(sampler_device->AllocWorkspace(
          sampler_ctx, num_input * fanout * sizeof(IdType)));
    }
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
    switch (RunConfig::sample_type) {
      case kKHop0:
        cuda::GPUSampleKHop0(indptr, indices, input, num_input, fanout, out_src,
                       out_dst, num_out, sampler_ctx, sample_stream,
                       random_states, task->key);
        break;
      case kKHop1:
        cuda::GPUSampleKHop1(indptr, indices, input, num_input, fanout, out_src,
                       out_dst, num_out, sampler_ctx, sample_stream,
                       random_states, task->key);
        break;
      case kWeightedKHop:
        cuda::GPUSampleWeightedKHop(indptr, indices, prob_table, alias_table, input,
                              num_input, fanout, out_src, out_dst, num_out,
                              sampler_ctx, sample_stream, random_states,
                              task->key);
        break;
      case kRandomWalk:
        CHECK_EQ(fanout, RunConfig::num_neighbor);
        cuda::GPUSampleRandomWalk(
            indptr, indices, input, num_input, RunConfig::random_walk_length,
            RunConfig::random_walk_restart_prob, RunConfig::num_random_walk,
            RunConfig::num_neighbor, out_src, out_dst, out_data, num_out,
            frequency_hashmap, sampler_ctx, sample_stream, random_states,
            task->key);
        break;
      case kWeightedKHopPrefix:
        cuda::GPUSampleWeightedKHopPrefix(indptr, indices, prob_prefix_table, input,
                              num_input, fanout, out_src, out_dst, num_out,
                              sampler_ctx, sample_stream, random_states,
                              task->key);
        break;
      case kKHop2:
        cuda::GPUSampleKHop2(indptr, const_cast<IdType*>(indices), input, num_input, fanout, out_src,
                       out_dst, num_out, sampler_ctx, sample_stream,
                       random_states, task->key);
        break;
      case kWeightedKHopHashDedup:
        cuda::GPUSampleWeightedKHopHashDedup(indptr, const_cast<IdType*>(indices), const_cast<float*>(prob_table), alias_table, input,
            num_input, fanout, out_src, out_dst, num_out, sampler_ctx, sample_stream, random_states, task->key);
        break;
      default:
        CHECK(0);
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
    IdType num_unique;

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

    cuda::GPUMapEdges(out_src, new_src, out_dst, new_dst, num_samples,
                hash_table->DeviceHandle(), sampler_ctx, sample_stream);

    double map_edges_time = t3.Passed();
    double remap_time = t1.Passed();
    double layer_time = tlayer.Passed();

    auto train_graph = std::make_shared<TrainGraph>();
    train_graph->num_src = num_unique;
    train_graph->num_dst = num_input;
    train_graph->num_edge = num_samples;
    train_graph->col = Tensor::FromBlob(
        new_src, DataType::kI32, {num_samples}, sampler_ctx,
        "train_graph.row_cuda_sample_" + std::to_string(task->key) + "_" +
            std::to_string(i));
    train_graph->row = Tensor::FromBlob(
        new_dst, DataType::kI32, {num_samples}, sampler_ctx,
        "train_graph.dst_cuda_sample_" + std::to_string(task->key) + "_" +
            std::to_string(i));
    if (out_data) {
      train_graph->data = Tensor::FromBlob(
          out_data, DataType::kI32, {num_samples}, sampler_ctx,
          "train_graph.dst_cuda_sample_" + std::to_string(task->key) + "_" +
              std::to_string(i));
    }

    task->graphs[i] = train_graph;

    total_num_samples += num_samples;

    // Do some clean jobs
    sampler_device->FreeWorkspace(sampler_ctx, out_src);
    sampler_device->FreeWorkspace(sampler_ctx, out_dst);
    sampler_device->FreeWorkspace(sampler_ctx, num_out);
    if (i == (int)last_layer_idx) {
        Profiler::Get().LogStep(task->key, kLogL2LastLayerTime,
                                   layer_time);
        Profiler::Get().LogStep(task->key, kLogL2LastLayerSize,
                                   num_unique);
    }
    Profiler::Get().LogStepAdd(task->key, kLogL2CoreSampleTime,
                               core_sample_time);
    Profiler::Get().LogStepAdd(task->key, kLogL2IdRemapTime, remap_time);
    Profiler::Get().LogStepAdd(task->key, kLogL3RemapPopulateTime,
                               populate_time);
    Profiler::Get().LogStepAdd(task->key, kLogL3RemapMapNodeTime, 0);
    Profiler::Get().LogStepAdd(task->key, kLogL3RemapMapEdgeTime,
                               map_edges_time);

    cur_input = Tensor::FromBlob(
        (void *)unique, DataType::kI32, {num_unique}, sampler_ctx,
        "cur_input_unique_cuda_" + std::to_string(task->key) + "_" +
            std::to_string(i));
    LOG(DEBUG) << "GPUSample: finish layer " << i;
  }

  task->input_nodes = cur_input;

  Profiler::Get().LogStep(task->key, kLogL1NumNode,
                          static_cast<double>(task->input_nodes->Shape()[0]));
  Profiler::Get().LogStep(task->key, kLogL1NumSample, total_num_samples);
  Profiler::Get().LogStepAdd(task->key, kLogL3RemapFillUniqueTime,
                             fill_unique_time);

  LOG(DEBUG) << "SampleLoop: process task with key " << task->key;
}

void DoGetCacheMissIndex(TaskPtr task) {
  // Get index of miss data and cache data
  // Timer t4;
  auto sampler_ctx = DistEngine::Get()->GetSamplerCtx();
  auto sampler_device = Device::Get(sampler_ctx);
  auto sample_stream = DistEngine::Get()->GetSampleStream();
  if (RunConfig::UseGPUCache() && DistEngine::Get()->IsInitialized()) {
    auto input_nodes =
        reinterpret_cast<const IdType *>(task->input_nodes->Data());
    const size_t num_input = task->input_nodes->Shape()[0];

    IdType *sampler_output_miss_src_index = static_cast<IdType *>(
        sampler_device->AllocWorkspace(sampler_ctx, sizeof(IdType) * num_input));
    IdType *sampler_output_miss_dst_index = static_cast<IdType *>(
        sampler_device->AllocWorkspace(sampler_ctx, sizeof(IdType) * num_input));
    IdType *sampler_output_cache_src_index = static_cast<IdType *>(
        sampler_device->AllocWorkspace(sampler_ctx, sizeof(IdType) * num_input));
    IdType *sampler_output_cache_dst_index = static_cast<IdType *>(
        sampler_device->AllocWorkspace(sampler_ctx, sizeof(IdType) * num_input));

    size_t num_output_miss;
    size_t num_output_cache;

    cuda::GetMissCacheIndex(
        DistEngine::Get()->GetCacheHashtable(), sampler_ctx,
        sampler_output_miss_src_index, sampler_output_miss_dst_index,
        &num_output_miss, sampler_output_cache_src_index,
        sampler_output_cache_dst_index, &num_output_cache, input_nodes, num_input,
        sample_stream);

    CHECK_EQ(num_output_miss + num_output_cache, num_input);

    auto dtype = task->input_nodes->Type();
    // To be freed in task queue after serialization
    task->miss_cache_index.miss_src_index =
        Tensor::FromBlob(sampler_output_miss_src_index, dtype,
                         {num_output_miss}, sampler_ctx, "miss_src_index");
    task->miss_cache_index.miss_dst_index =
        Tensor::FromBlob(sampler_output_miss_dst_index, dtype,
                         {num_output_miss}, sampler_ctx, "miss_dst_index");
    task->miss_cache_index.cache_src_index =
        Tensor::FromBlob(sampler_output_cache_src_index, dtype,
                         {num_output_cache}, sampler_ctx, "cache_src_index");
    task->miss_cache_index.cache_dst_index =
        Tensor::FromBlob(sampler_output_cache_dst_index, dtype,
                         {num_output_cache}, sampler_ctx, "cache_dst_index");

    sampler_device->StreamSync(sampler_ctx, sample_stream);

    task->miss_cache_index.num_miss = num_output_miss;
    task->miss_cache_index.num_cache = num_output_cache;
  }
}

void DoGraphCopy(TaskPtr task) {
  auto trainer_ctx = DistEngine::Get()->GetTrainerCtx();
  auto copy_ctx = trainer_ctx;
  auto copy_device = Device::Get(trainer_ctx);
  auto copy_stream = DistEngine::Get()->GetTrainerCopyStream();

  for (size_t i = 0; i < task->graphs.size(); i++) {
    auto graph = task->graphs[i];
    if (graph->row->Ctx() == trainer_ctx) {
      CHECK(graph->col->Ctx() == trainer_ctx) << "col ctx needs equal row in graph";
      continue;
    }
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

    copy_device->CopyDataFromTo(graph->row->Data(), 0,
                                   train_row->MutableData(), 0,
                                   graph->row->NumBytes(), graph->row->Ctx(),
                                   train_row->Ctx(), copy_stream);
    copy_device->CopyDataFromTo(graph->col->Data(), 0,
                                   train_col->MutableData(), 0,
                                   graph->col->NumBytes(), graph->col->Ctx(),
                                   train_col->Ctx(), copy_stream);

    if (RunConfig::sample_type == kRandomWalk) {
      auto graph_data = Tensor::Empty(
          graph->data->Type(), graph->data->Shape(), trainer_ctx,
          "train_graph.data_cuda_train_" + std::to_string(task->key) + "_" +
              std::to_string(i));

      LOG(DEBUG) << "GraphCopyDevice2DeviceLoop: cuda graph data malloc "
                 << ToReadableSize(graph->data->NumBytes());

      copy_device->CopyDataFromTo(
          graph->data->Data(), 0, graph_data->MutableData(), 0,
          graph->data->NumBytes(), graph->data->Ctx(), graph_data->Ctx(),
          copy_stream);
      graph->data = graph_data;
    }

    copy_device->StreamSync(copy_ctx, copy_stream);

    graph->row = train_row;
    graph->col = train_col;

    Profiler::Get().LogStepAdd(task->key, kLogL1GraphBytes,
                               train_row->NumBytes() + train_col->NumBytes());
  }

  LOG(DEBUG) << "GraphCopyDevice2Device: process task with key " << task->key;
}

void DoIdCopy(TaskPtr task) {
  auto copy_ctx = task->input_nodes->Ctx();
  auto copy_device = Device::Get(copy_ctx);
  auto copy_stream = DistEngine::Get()->GetSamplerCopyStream();

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

  copy_device->CopyDataFromTo(
      task->input_nodes->Data(), 0, input_nodes->MutableData(), 0,
      task->input_nodes->NumBytes(), task->input_nodes->Ctx(),
      input_nodes->Ctx(), copy_stream);
  copy_device->CopyDataFromTo(
      task->output_nodes->Data(), 0, output_nodes->MutableData(), 0,
      task->output_nodes->NumBytes(), task->output_nodes->Ctx(),
      output_nodes->Ctx(), copy_stream);

  copy_device->StreamSync(copy_ctx, copy_stream);

  task->input_nodes = input_nodes;
  task->output_nodes = output_nodes;

  Profiler::Get().LogStepAdd(
      task->key, kLogL1IdBytes,
      input_nodes->NumBytes() + output_nodes->NumBytes());

  LOG(DEBUG) << "IdCopyDevice2Host: process task with key " << task->key;
}

void DoCPUFeatureExtract(TaskPtr task) {
  auto dataset = DistEngine::Get()->GetGraphDataset();

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

  LOG(DEBUG) << "DoCPUFeatureExtract input_feat cpu malloc "
             << ToReadableSize(task->input_feat->NumBytes());
  LOG(DEBUG) << "DoCPUFeatureExtract output_label cpu malloc "
             << ToReadableSize(task->output_label->NumBytes());

  auto feat_dst = task->input_feat->MutableData();
  auto feat_src = dataset->feat->Data();

  if (RunConfig::option_empty_feat != 0) {
    cpu::CPUMockExtract(feat_dst, feat_src, input_data, num_input, feat_dim,
                    feat_type);
  } else {
    cpu::CPUExtract(feat_dst, feat_src, input_data, num_input, feat_dim,
                    feat_type);
  }

  auto label_dst = task->output_label->MutableData();
  auto label_src = dataset->label->Data();

  cpu::CPUExtract(label_dst, label_src, output_data, num_ouput, 1, label_type);

  if (RunConfig::option_log_node_access) {
    Profiler::Get().LogNodeAccess(task->key, input_data, num_input);
  }

  LOG(DEBUG) << "HostFeatureExtract: process task with key " << task->key;
}

void DoFeatureCopy(TaskPtr task) {
  auto trainer_ctx = DistEngine::Get()->GetTrainerCtx();
  auto trainer_device = Device::Get(trainer_ctx);
  auto copy_stream = DistEngine::Get()->GetTrainerCopyStream();

  auto cpu_feat = task->input_feat;
  auto cpu_label = task->output_label;

  CHECK_EQ(cpu_feat->Ctx().device_type, CPU().device_type);
  CHECK_EQ(cpu_label->Ctx().device_type, CPU().device_type);

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

  Profiler::Get().LogStep(task->key, kLogL1FeatureBytes,
                          train_feat->NumBytes());
  Profiler::Get().LogStep(task->key, kLogL1LabelBytes, train_label->NumBytes());

  Profiler::Get().LogEpochAdd(task->key, kLogEpochFeatureBytes,
                              train_feat->NumBytes());
  Profiler::Get().LogEpochAdd(task->key, kLogEpochMissBytes,
                              train_feat->NumBytes());

  LOG(DEBUG) << "FeatureCopyHost2Device: process task with key " << task->key;
}

void DoCacheIdCopy(TaskPtr task) {
  auto trainer_ctx = DistEngine::Get()->GetTrainerCtx();
  auto copy_ctx = trainer_ctx;
  auto copy_device = Device::Get(trainer_ctx);
  auto copy_stream = DistEngine::Get()->GetTrainerCopyStream();

  auto output_nodes = Tensor::Empty(
      task->output_nodes->Type(), task->output_nodes->Shape(), trainer_ctx,
      "task.output_nodes_cuda_" + std::to_string(task->key));
  LOG(DEBUG) << "IdCopyHost2Device output_nodes cuda malloc "
             << ToReadableSize(output_nodes->NumBytes());

  copy_device->CopyDataFromTo(
      task->output_nodes->Data(), 0, output_nodes->MutableData(), 0,
      task->output_nodes->NumBytes(), task->output_nodes->Ctx(),
      output_nodes->Ctx(), copy_stream);

  copy_device->StreamSync(copy_ctx, copy_stream);

  task->output_nodes = output_nodes;

  Profiler::Get().LogStepAdd(task->key, kLogL1IdBytes,
                             output_nodes->NumBytes());

  LOG(DEBUG) << "IdCopyDevice2Host: process task with key " << task->key;
}

void DoCacheIdCopyToCPU(TaskPtr task) {
  auto sampler_ctx = DistEngine::Get()->GetSamplerCtx();
  auto sampler_device = Device::Get(sampler_ctx);
  auto copy_stream = DistEngine::Get()->GetSamplerCopyStream();

  auto output_nodes = Tensor::Empty(
      task->output_nodes->Type(), task->output_nodes->Shape(), CPU(),
      "task.output_nodes_cuda_" + std::to_string(task->key));
  LOG(DEBUG) << "IdCopyDevice2Host output_nodes cuda malloc "
             << ToReadableSize(output_nodes->NumBytes());

  sampler_device->CopyDataFromTo(
      task->output_nodes->Data(), 0, output_nodes->MutableData(), 0,
      task->output_nodes->NumBytes(), task->output_nodes->Ctx(),
      output_nodes->Ctx(), copy_stream);

  sampler_device->StreamSync(sampler_ctx, copy_stream);

  task->output_nodes = output_nodes;

  Profiler::Get().LogStepAdd(task->key, kLogL1IdBytes,
                             output_nodes->NumBytes());

  LOG(DEBUG) << "IdCopyDevice2Host: process task with key " << task->key;
}

// for switcher cache
void DoSwitchCacheFeatureCopy(TaskPtr task) {
  auto trainer_ctx = DistEngine::Get()->GetTrainerCtx();
  auto trainer_device = Device::Get(trainer_ctx);
  auto trainer_copy_stream = DistEngine::Get()->GetTrainerCopyStream();
  auto cpu_ctx = CPU();
  auto cpu_device = Device::Get(cpu_ctx);

  auto dataset = DistEngine::Get()->GetGraphDataset();
  auto cache_manager = DistEngine::Get()->GetGPUCacheManager();

  auto input_nodes = task->input_nodes;
  auto feat = dataset->feat;
  auto feat_dim = dataset->feat->Shape()[1];
  auto feat_type = dataset->feat->Type();

  auto input_data = reinterpret_cast<const IdType *>(input_nodes->Data());
  auto num_input = input_nodes->Shape()[0];

  auto train_feat =
      Tensor::Empty(feat_type, {num_input, feat_dim}, trainer_ctx,
                    "task.train_feat_cuda_" + std::to_string(task->key));

  // 1. Get index of miss data and cache data
  // feature data has cache, so we only need to extract the miss data
  // do get_miss_cache_index in trainer GPU
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
      trainer_copy_stream);

  CHECK_EQ(num_output_miss + num_output_cache, num_input);

  double get_index_time = t0.Passed();

  // 2. Move the miss index

  Timer t1;

  IdType *cpu_output_miss_src_index = static_cast<IdType *>(
      cpu_device->AllocWorkspace(CPU(), sizeof(IdType) * num_output_miss));
  trainer_device->CopyDataFromTo(trainer_output_miss_src_index, 0,
                                 cpu_output_miss_src_index, 0,
                                 num_output_miss * sizeof(IdType), trainer_ctx,
                                 CPU(), trainer_copy_stream);
  trainer_device->StreamSync(trainer_ctx, trainer_copy_stream);
  // free the miss src indices in GPU
  trainer_device->FreeWorkspace(trainer_ctx, trainer_output_miss_src_index);

  double copy_idx_time = t1.Passed();

  // 3. Extract the miss data
  Timer t2;

  void *cpu_output_miss = cpu_device->AllocWorkspace(
      CPU(), GetTensorBytes(feat_type, {num_output_miss, feat_dim}));
  void *trainer_output_miss = trainer_device->AllocWorkspace(
      trainer_ctx, GetTensorBytes(feat_type, {num_output_miss, feat_dim}));

  cache_manager->ExtractMissData(cpu_output_miss, cpu_output_miss_src_index,
                                 num_output_miss);

  double extract_miss_time = t2.Passed();

  // 4. Copy the miss data
  Timer t3;

  trainer_device->CopyDataFromTo(
      cpu_output_miss, 0, trainer_output_miss, 0,
      GetTensorBytes(feat_type, {num_output_miss, feat_dim}), CPU(),
      trainer_ctx, trainer_copy_stream);
  trainer_device->StreamSync(trainer_ctx, trainer_copy_stream);

  cpu_device->FreeWorkspace(CPU(), cpu_output_miss);

  double copy_miss_time = t3.Passed();

  // 5. Combine miss data
  Timer t4;
  cache_manager->CombineMissData(train_feat->MutableData(), trainer_output_miss,
                                 trainer_output_miss_dst_index, num_output_miss,
                                 trainer_copy_stream);
  trainer_device->StreamSync(trainer_ctx, trainer_copy_stream);

  double combine_miss_time = t4.Passed();

  // 6. Combine cache data
  Timer t5;
  cache_manager->CombineCacheData(
      train_feat->MutableData(), trainer_output_cache_src_index,
      trainer_output_cache_dst_index, num_output_cache, trainer_copy_stream);
  trainer_device->StreamSync(trainer_ctx, trainer_copy_stream);

  double combine_cache_time = t5.Passed();

  task->input_feat = train_feat;

  // 7. Free space
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
  Profiler::Get().LogStep(task->key, kLogL3CacheGetIndexTime, get_index_time);
  Profiler::Get().LogStep(task->key, KLogL3CacheCopyIndexTime, copy_idx_time);
  Profiler::Get().LogStep(task->key, kLogL3CacheExtractMissTime,
                          extract_miss_time);
  Profiler::Get().LogStep(task->key, kLogL3CacheCopyMissTime, copy_miss_time);
  Profiler::Get().LogStep(task->key, kLogL3CacheCombineMissTime,
                          combine_miss_time);
  Profiler::Get().LogStep(task->key, kLogL3CacheCombineCacheTime,
                          combine_cache_time);
  Profiler::Get().LogEpochAdd(task->key, kLogEpochFeatureBytes,
                              train_feat->NumBytes());
  Profiler::Get().LogEpochAdd(
      task->key, kLogEpochMissBytes,
      GetTensorBytes(feat_type, {num_output_miss, feat_dim}));

  LOG(DEBUG) << "DoSwitchCacheFeatureCopy: process task with key " << task->key;
}

void DoCacheFeatureCopy(TaskPtr task) {
  auto trainer_ctx = DistEngine::Get()->GetTrainerCtx();
  auto trainer_device = Device::Get(trainer_ctx);
  auto trainer_copy_stream = DistEngine::Get()->GetTrainerCopyStream();
  auto cpu_ctx = CPU();
  auto cpu_device = Device::Get(cpu_ctx);

  auto dataset = DistEngine::Get()->GetGraphDataset();
  auto cache_manager = DistEngine::Get()->GetCacheManager();

  auto input_nodes = task->input_nodes;
  auto feat = dataset->feat;
  auto feat_dim = dataset->feat->Shape()[1];
  auto feat_type = dataset->feat->Type();

  auto num_input =
      task->miss_cache_index.num_cache + task->miss_cache_index.num_miss;

  auto train_feat =
      Tensor::Empty(feat_type, {num_input, feat_dim}, trainer_ctx,
                    "task.train_feat_cuda_" + std::to_string(task->key));

  // 1. Get index of miss data and cache data
  // feature data has cache, so we only need to extract the miss data
  Timer t0;

  size_t num_output_miss = task->miss_cache_index.num_miss;
  size_t num_output_cache = task->miss_cache_index.num_cache;

  const IdType *output_miss_src_index = nullptr;
  const IdType *trainer_output_miss_dst_index = nullptr;
  const IdType *trainer_output_cache_src_index = nullptr;
  const IdType *trainer_output_cache_dst_index = nullptr;
  if (num_output_miss > 0) {
    output_miss_src_index = static_cast<const IdType *>(
        task->miss_cache_index.miss_src_index->Data());
    CHECK_EQ(task->miss_cache_index.miss_dst_index->Ctx(), trainer_ctx)
      << "output_miss_dst_index should be in trainer GPU";
    trainer_output_miss_dst_index = static_cast<const IdType *>(
        task->miss_cache_index.miss_dst_index->Data());
  }

  if (num_output_cache > 0) {
    CHECK_EQ(task->miss_cache_index.cache_src_index->Ctx(), trainer_ctx)
      << "output_cache_src_index should be in trainer GPU";
    trainer_output_cache_src_index = static_cast<const IdType *>(
        task->miss_cache_index.cache_src_index->Data());
    CHECK_EQ(task->miss_cache_index.cache_dst_index->Ctx(), trainer_ctx)
      << "output_cache_dst_index should be in trainer GPU";
    trainer_output_cache_dst_index = static_cast<const IdType *>(
        task->miss_cache_index.cache_dst_index->Data());
  }

  double get_index_time = t0.Passed();

  // 2. Move the miss index

  Timer t1;

  const IdType *cpu_output_miss_src_index = output_miss_src_index;

  double copy_idx_time = t1.Passed();

  // 3. Extract and copy the miss data
  Timer t2;

  void *cpu_output_miss = cpu_device->AllocWorkspace(
      CPU(), GetTensorBytes(feat_type, {num_output_miss, feat_dim}));
  void *trainer_output_miss = trainer_device->AllocWorkspace(
      trainer_ctx, GetTensorBytes(feat_type, {num_output_miss, feat_dim}));

  cache_manager->ExtractMissData(cpu_output_miss, cpu_output_miss_src_index,
                                 num_output_miss);

  double extract_miss_time = t2.Passed();

  // 4. Copy the miss data
  Timer t3;

  trainer_device->CopyDataFromTo(
      cpu_output_miss, 0, trainer_output_miss, 0,
      GetTensorBytes(feat_type, {num_output_miss, feat_dim}), cpu_ctx,
      trainer_ctx, trainer_copy_stream);
  trainer_device->StreamSync(trainer_ctx, trainer_copy_stream);

  cpu_device->FreeWorkspace(CPU(), cpu_output_miss);

  double copy_miss_time = t3.Passed();

  // 5. Combine miss data
  Timer t4;
  cache_manager->CombineMissData(train_feat->MutableData(), trainer_output_miss,
                                 trainer_output_miss_dst_index, num_output_miss,
                                 trainer_copy_stream);
  trainer_device->StreamSync(trainer_ctx, trainer_copy_stream);

  double combine_miss_time = t4.Passed();

  // 6. Combine cache data
  Timer t5;
  cache_manager->CombineCacheData(
      train_feat->MutableData(), trainer_output_cache_src_index,
      trainer_output_cache_dst_index, num_output_cache, trainer_copy_stream);
  trainer_device->StreamSync(trainer_ctx, trainer_copy_stream);

  double combine_cache_time = t5.Passed();

  task->input_feat = train_feat;

  // 7. Free space
  trainer_device->FreeWorkspace(trainer_ctx, trainer_output_miss);

  Profiler::Get().LogStep(task->key, kLogL1FeatureBytes,
                          train_feat->NumBytes());
  Profiler::Get().LogStep(
      task->key, kLogL1MissBytes,
      GetTensorBytes(feat_type, {num_output_miss, feat_dim}));
  Profiler::Get().LogStep(task->key, kLogL3CacheGetIndexTime, get_index_time);
  Profiler::Get().LogStep(task->key, KLogL3CacheCopyIndexTime, copy_idx_time);
  Profiler::Get().LogStep(task->key, kLogL3CacheExtractMissTime,
                          extract_miss_time);
  Profiler::Get().LogStep(task->key, kLogL3CacheCopyMissTime, copy_miss_time);
  Profiler::Get().LogStep(task->key, kLogL3CacheCombineMissTime,
                          combine_miss_time);
  Profiler::Get().LogStep(task->key, kLogL3CacheCombineCacheTime,
                          combine_cache_time);
  Profiler::Get().LogEpochAdd(task->key, kLogEpochFeatureBytes,
                              train_feat->NumBytes());
  Profiler::Get().LogEpochAdd(
      task->key, kLogEpochMissBytes,
      GetTensorBytes(feat_type, {num_output_miss, feat_dim}));

  LOG(DEBUG) << "DoCacheFeatureCopy: process task with key " << task->key;
}

void DoGPULabelExtract(TaskPtr task) {
  auto dataset = DistEngine::Get()->GetGraphDataset();

  auto trainer_ctx = DistEngine::Get()->GetTrainerCtx();
  auto trainer_device = Device::Get(trainer_ctx);
  auto trainer_copy_stream = DistEngine::Get()->GetTrainerCopyStream();

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
             trainer_ctx, trainer_copy_stream, task->key);

  trainer_device->StreamSync(trainer_ctx, trainer_copy_stream);

  LOG(DEBUG) << "HostFeatureExtract: process task with key " << task->key;

  task->output_label = train_label;

  Profiler::Get().LogStep(task->key, kLogL1LabelBytes, train_label->NumBytes());
}

void DoCPULabelExtractAndCopy(TaskPtr task) {
  auto dataset = DistEngine::Get()->GetGraphDataset();
  auto trainer_ctx = DistEngine::Get()->GetTrainerCtx();
  auto trainer_device = Device::Get(trainer_ctx);
  auto copy_stream = DistEngine::Get()->GetTrainerCopyStream();

  // 1. Extract
  auto output_nodes = task->output_nodes;

  auto label = dataset->label;
  auto label_type = dataset->label->Type();

  auto output_data = reinterpret_cast<const IdType *>(output_nodes->Data());
  auto num_ouput = output_nodes->Shape()[0];

  task->output_label =
      Tensor::Empty(label_type, {num_ouput}, CPU(),
                    "task.output_label_cpu" + std::to_string(task->key));

  LOG(DEBUG) << "DoCPUFeatureExtract output_label cpu malloc "
             << ToReadableSize(task->output_label->NumBytes());

  auto label_dst = task->output_label->MutableData();
  auto label_src = dataset->label->Data();

  cpu::CPUExtract(label_dst, label_src, output_data, num_ouput, 1, label_type);

  // 2. Copy
  auto cpu_label = task->output_label;
  CHECK_EQ(cpu_label->Ctx().device_type, CPU().device_type);
  auto train_label =
      Tensor::Empty(cpu_label->Type(), cpu_label->Shape(), trainer_ctx,
                    "task.train_label_cuda" + std::to_string(task->key));
  trainer_device->CopyDataFromTo(
      cpu_label->Data(), 0, train_label->MutableData(), 0,
      cpu_label->NumBytes(), cpu_label->Ctx(), train_label->Ctx(), copy_stream);
  trainer_device->StreamSync(trainer_ctx, copy_stream);

  task->output_label = train_label;

  Profiler::Get().LogStep(task->key, kLogL1LabelBytes, train_label->NumBytes());

  LOG(DEBUG) << "DoCPULabelExtractAndCopy: process task with key " << task->key;
}

void DoArch6GetCacheMissIndex(TaskPtr task) {
  auto trainer_ctx = DistEngine::Get()->GetTrainerCtx();
  auto trainer_device = Device::Get(trainer_ctx);
  auto stream = DistEngine::Get()->GetTrainerCopyStream();

  auto dataset = DistEngine::Get()->GetGraphDataset();
  auto cache_manager = DistEngine::Get()->GetGPUCacheManager();

  auto input_nodes = task->input_nodes;
  auto feat = dataset->feat;

  auto input_data = reinterpret_cast<const IdType *>(input_nodes->Data());
  auto num_input = input_nodes->Shape()[0];

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

  auto dtype = task->input_nodes->Type();
  // To be freed in task queue after serialization
  task->miss_cache_index.miss_src_index =
      Tensor::FromBlob(trainer_output_miss_src_index, dtype, {num_output_miss},
                       trainer_ctx, "miss_src_index");
  task->miss_cache_index.miss_dst_index =
      Tensor::FromBlob(trainer_output_miss_dst_index, dtype, {num_output_miss},
                       trainer_ctx, "miss_dst_index");
  task->miss_cache_index.cache_src_index =
      Tensor::FromBlob(trainer_output_cache_src_index, dtype,
                       {num_output_cache}, trainer_ctx, "cache_src_index");
  task->miss_cache_index.cache_dst_index =
      Tensor::FromBlob(trainer_output_cache_dst_index, dtype,
                       {num_output_cache}, trainer_ctx, "cache_dst_index");

  trainer_device->StreamSync(trainer_ctx, stream);

  task->miss_cache_index.num_miss = num_output_miss;
  task->miss_cache_index.num_cache = num_output_cache;
}

void DoArch6CacheFeatureCopy(TaskPtr task) {
  auto trainer_ctx = DistEngine::Get()->GetTrainerCtx();
  auto trainer_device = Device::Get(trainer_ctx);
  auto cpu_device = Device::Get(CPU());
  auto stream = DistEngine::Get()->GetTrainerCopyStream();

  auto dataset = DistEngine::Get()->GetGraphDataset();
  auto cache_manager = DistEngine::Get()->GetGPUCacheManager();

  auto input_nodes = task->input_nodes;
  auto feat = dataset->feat;
  auto feat_dim = dataset->feat->Shape()[1];
  auto feat_type = dataset->feat->Type();

  auto num_input = input_nodes->Shape()[0];

  CHECK_EQ(input_nodes->Ctx().device_type, trainer_ctx.device_type);
  CHECK_EQ(input_nodes->Ctx().device_id, trainer_ctx.device_id);

  auto train_feat =
      Tensor::Empty(feat_type, {num_input, feat_dim}, trainer_ctx,
                    "task.train_feat_cuda_" + std::to_string(task->key));

  // 0. Get index of miss data and cache data
  // feature data has cache, so we only need to extract the miss data
  Timer t0;

  size_t num_output_miss = task->miss_cache_index.num_miss;
  size_t num_output_cache = task->miss_cache_index.num_cache;

  IdType *trainer_output_miss_src_index = static_cast<IdType *>(
      task->miss_cache_index.miss_src_index->MutableData());
  IdType *trainer_output_miss_dst_index = static_cast<IdType *>(
      task->miss_cache_index.miss_dst_index->MutableData());
  IdType *trainer_output_cache_src_index = static_cast<IdType *>(
      task->miss_cache_index.cache_src_index->MutableData());
  IdType *trainer_output_cache_dst_index = static_cast<IdType *>(
      task->miss_cache_index.cache_dst_index->MutableData());

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
  task->miss_cache_index.miss_src_index = nullptr;
  task->miss_cache_index.miss_dst_index = nullptr;
  task->miss_cache_index.cache_src_index = nullptr;
  task->miss_cache_index.cache_dst_index = nullptr;

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

} // dist
} // common
} // samgraph
