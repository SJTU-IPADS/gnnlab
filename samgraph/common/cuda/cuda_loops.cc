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

#include <fstream>

#include "cuda_loops.h"

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

TaskPtr DoShuffle() {
  auto s = GPUEngine::Get()->GetShuffler();
  auto sample_stream = GPUEngine::Get()->GetSampleStream();
  auto batch = s->GetBatch(sample_stream);

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

  auto random_states = GPUEngine::Get()->GetRandomStates();
  auto frequency_hashmap = GPUEngine::Get()->GetFrequencyHashmap();

  OrderedHashTable *hash_table = GPUEngine::Get()->GetHashtable();
  hash_table->Reset(sample_stream);

  Timer t;
  auto output_nodes = task->output_nodes;
  size_t num_train_node = output_nodes->Shape()[0];
  hash_table->FillWithUnique(output_nodes->CPtr<IdType>(), num_train_node,
      sample_stream);
  task->graphs.resize(num_layers);
  double fill_unique_time = t.Passed();

  const IdType *indptr = dataset->indptr->CPtr<IdType>();
  const IdType *indices = dataset->indices->CPtr<IdType>();
  const float *prob_table = dataset->prob_table->CPtr<float>();
  const IdType *alias_table = dataset->alias_table->CPtr<IdType>();
  const float *prob_prefix_table = dataset->prob_prefix_table->CPtr<float>();

  auto cur_input = task->output_nodes;
  size_t total_num_samples = 0;

  for (int i = last_layer_idx; i >= 0; i--) {
    Timer tlayer;
    Timer t0;
    const size_t fanout = fanouts[i];
    const IdType *input = cur_input->CPtr<IdType>();
    const size_t num_input = cur_input->Shape()[0];
    LOG(DEBUG) << "DoGPUSample: begin sample layer=" << i 
               << " ctx=" << sampler_ctx
               << " num_input=" << num_input;

    IdType *out_src = sampler_device->AllocArray<IdType>(sampler_ctx, num_input * fanout);
    IdType *out_dst = sampler_device->AllocArray<IdType>(sampler_ctx, num_input * fanout);
    IdType *out_data = nullptr;
    if (RunConfig::sample_type == kRandomWalk) {
      out_data = sampler_device->AllocArray<IdType>(sampler_ctx, num_input * fanout);
    }
    size_t *num_out = sampler_device->AllocArray<size_t>(sampler_ctx, 1);
    size_t num_samples;

    LOG(DEBUG) << "DoGPUSample: size of out_src " << num_input * fanout;
    LOG(TRACE) << "DoGPUSample: cuda out_src malloc "
               << ToReadableSize(num_input * fanout * sizeof(IdType));
    LOG(TRACE) << "DoGPUSample: cuda out_dst malloc "
               << ToReadableSize(num_input * fanout * sizeof(IdType));
    LOG(TRACE) << "DoGPUSample: cuda num_out malloc "
               << ToReadableSize(sizeof(size_t));

    // Sample a compact coo graph
    switch (RunConfig::sample_type) {
      case kKHop0:
        GPUSampleKHop0(indptr, indices, input, num_input, fanout, out_src,
                      out_dst, num_out, sampler_ctx, sample_stream,
                      random_states, task->key);
        break;
      case kKHop1:
        GPUSampleKHop1(indptr, indices, input, num_input, fanout, out_src,
                      out_dst, num_out, sampler_ctx, sample_stream,
                      random_states, task->key);
        break;
      case kWeightedKHop:
        GPUSampleWeightedKHop(indptr, indices, prob_table, alias_table, input,
                              num_input, fanout, out_src, out_dst, num_out,
                              sampler_ctx, sample_stream, random_states,
                              task->key);
        break;
      case kRandomWalk:
        CHECK_EQ(fanout, RunConfig::num_neighbor);
        GPUSampleRandomWalk(
            indptr, indices, input, num_input, RunConfig::random_walk_length,
            RunConfig::random_walk_restart_prob, RunConfig::num_random_walk,
            RunConfig::num_neighbor, out_src, out_dst, out_data, num_out,
            frequency_hashmap, sampler_ctx, sample_stream, random_states,
            task->key);
        break;
      case kWeightedKHopPrefix:
        GPUSampleWeightedKHopPrefix(indptr, indices, prob_prefix_table, input,
                              num_input, fanout, out_src, out_dst, num_out,
                              sampler_ctx, sample_stream, random_states,
                              task->key);
        break;
      case kKHop2:
        GPUSampleKHop2(indptr, const_cast<IdType*>(indices), input, num_input, fanout, out_src,
                      out_dst, num_out, sampler_ctx, sample_stream,
                      random_states, task->key);
        break;
      case kWeightedKHopHashDedup:
        GPUSampleWeightedKHopHashDedup(indptr, const_cast<IdType*>(indices), const_cast<float*>(prob_table), alias_table, input,
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
    IdType *unique = sampler_device->AllocArray<IdType>(
        sampler_ctx, (num_samples + hash_table->NumItems()));
    IdType num_unique;

    LOG(DEBUG) << "GPUSample: cuda unique malloc "
               << ToReadableSize((num_samples + +hash_table->NumItems()) *
                                 sizeof(IdType));

    hash_table->FillWithDuplicates(out_dst, num_samples, unique, &num_unique,
                                   sample_stream);

    double populate_time = t2.Passed();

    Timer t3;

    // Mapping edges
    IdType *new_src = sampler_device->AllocArray<IdType>(sampler_ctx, num_samples);
    IdType *new_dst = sampler_device->AllocArray<IdType>(sampler_ctx, num_samples);

    LOG(DEBUG) << "GPUSample: size of new_src " << num_samples;
    LOG(TRACE) << "GPUSample: cuda new_src malloc "
               << ToReadableSize(num_samples * sizeof(IdType));
    LOG(TRACE) << "GPUSample: cuda new_dst malloc "
               << ToReadableSize(num_samples * sizeof(IdType));

    GPUMapEdges(out_src, new_src, out_dst, new_dst, num_samples,
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
    LOG(DEBUG) << "_debug task_key=" << task->key << " layer=" << i << " ctx=" << sampler_ctx
               << " num_unique=" << num_unique << " num_sample=" << num_samples
               << " remap_time ( " << remap_time << " )"
               << " = populate_time ( " << populate_time << " )"
               << " + edge_map_time ( " << map_edges_time << " )";
    Profiler::Get().LogStepAdd(task->key, kLogL2CoreSampleTime,
                               core_sample_time);
    // LOG(WARNING) << "sam.kLogEpochCoreSampleTime" << kLogEpochCoreSampleTime;
    Profiler::Get().LogEpochAdd(task->key, kLogEpochCoreSampleTime, core_sample_time);
    Profiler::Get().LogStepAdd(task->key, kLogL2IdRemapTime, remap_time);
    Profiler::Get().LogEpochAdd(task->key, kLogEpochIdRemapTime, remap_time);
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
                          static_cast<double>(cur_input->Shape()[0]));
  Profiler::Get().LogStep(task->key, kLogL1NumSample, total_num_samples);
  Profiler::Get().LogStepAdd(task->key, kLogL3RemapFillUniqueTime,
                             fill_unique_time);

  LOG(DEBUG) << "SampleLoop: process task with key " << task->key;
}

void DoGPUSampleDyCache(TaskPtr task, std::function<void(TaskPtr)> & nbr_cb) {
  auto fanouts = GPUEngine::Get()->GetFanout();
  auto num_layers = fanouts.size();
  auto last_layer_idx = num_layers - 1;

  auto dataset = GPUEngine::Get()->GetGraphDataset();
  auto sampler_ctx = GPUEngine::Get()->GetSamplerCtx();
  auto sampler_device = Device::Get(sampler_ctx);
  auto sample_stream = GPUEngine::Get()->GetSampleStream();

  auto random_states = GPUEngine::Get()->GetRandomStates();

  OrderedHashTable *hash_table = GPUEngine::Get()->GetHashtable();
  hash_table->Reset(sample_stream);

  Timer t;
  auto output_nodes = task->output_nodes;
  size_t num_train_node = output_nodes->Shape()[0];
  hash_table->FillWithUnique(
      output_nodes->CPtr<IdType>(), num_train_node, sample_stream);
  task->graphs.resize(num_layers);
  double fill_unique_time = t.Passed();

  const IdType *indptr = dataset->indptr->CPtr<IdType>();
  const IdType *indices = dataset->indices->CPtr<IdType>();
  const float *prob_table = dataset->prob_table->CPtr<float>();
  const IdType *alias_table = dataset->alias_table->CPtr<IdType>();

  const IdType* input = task->output_nodes->Ptr<IdType>();
  size_t num_input = task->output_nodes->Shape()[0];

  Timer prefetc_improved;
  double get_neighbour_time;
  for (int i = last_layer_idx; i >= 0; i--) {
    Timer tlayer;
    Timer t0;
    const size_t fanout = fanouts[i];
    LOG(DEBUG) << "DoGPUSample: begin sample layer " << i;

    IdType *out_src = sampler_device->AllocArray<IdType>(sampler_ctx, num_input * fanout);
    IdType *out_dst = sampler_device->AllocArray<IdType>(sampler_ctx, num_input * fanout);
    size_t *num_out = sampler_device->AllocArray<size_t>(sampler_ctx, 1);
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
        GPUSampleKHop0(indptr, indices, input, num_input, fanout, out_src,
                       out_dst, num_out, sampler_ctx, sample_stream,
                       random_states, task->key);
        break;
      case kKHop1:
        GPUSampleKHop1(indptr, indices, input, num_input, fanout, out_src,
                       out_dst, num_out, sampler_ctx, sample_stream,
                       random_states, task->key);
        break;
      case kWeightedKHop:
        GPUSampleWeightedKHop(indptr, indices, prob_table, alias_table, input,
                              num_input, fanout, out_src, out_dst, num_out,
                              sampler_ctx, sample_stream, random_states,
                              task->key);
        break;
      case kRandomWalk:
      case kWeightedKHopPrefix:
        // to ensure all neighbour of last layer covers sampled nodes
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

    LOG(DEBUG) << "GPUSample: cuda unique malloc "
               << ToReadableSize((num_samples + +hash_table->NumItems()) *
                                 sizeof(IdType));

    IdType num_unique;
    const IdType *unique;

    // if (i == 0) {
    //   // last layer, no need to put into hash table again
    //   hash_table->RefUnique(unique, &num_unique);
    // } else if (i == 1) {
    //   // 2nd last layer
    //   hash_table->FillWithDupRevised(out_dst, num_samples,
    //                                 sample_stream);
    //   hash_table->RefUnique(unique, &num_unique);
    //   hash_table->FillNeighbours(indptr, indices, sample_stream);
    //   IdType num_unique;
    //   const IdType *unique;
    //   hash_table->RefUnique(unique, &num_unique);
    //   task->input_nodes = Tensor::CopyBlob(
    //       unique, DataType::kI32, {num_unique}, sampler_ctx, sampler_ctx, 
    //       "cur_input_unique_cuda_" + std::to_string(task->key) + "_0");
    // } else 
    // {
    //   hash_table->FillWithDupRevised(out_dst, num_samples,
    //                                 sample_stream);
    //   hash_table->RefUnique(unique, &num_unique);
    // }

    if (i == 0) {
      // // last layer, no need to put into hash table again
      // hash_table->FillNeighbours(indptr, indices, sample_stream);
      // hash_table->RefUnique(unique, &num_unique);
      hash_table->RefUnique(unique, &num_unique);
    } else if (i == 1) {
      // 2nd last layer
      hash_table->FillWithDupRevised(out_dst, num_samples,
                                    sample_stream);
      hash_table->RefUnique(unique, &num_unique);
      {
        Timer tt;
        IdType *nbrs;
        size_t num_nbrs_dup;
        GPUExtractNeighbour(indptr, indices, unique, num_unique, nbrs, &num_nbrs_dup, sampler_ctx, sample_stream, task->key);
        hash_table->FillWithDupMutable(nbrs, num_nbrs_dup, sample_stream);
        sampler_device->FreeWorkspace(sampler_ctx, nbrs);
        const IdType * ids_prefetch;
        IdType n_ids_prefetch;
        hash_table->RefUnique(ids_prefetch, &n_ids_prefetch);
        task->input_nodes = Tensor::CopyBlob(
            ids_prefetch, DataType::kI32, {n_ids_prefetch}, sampler_ctx, sampler_ctx, 
            "cur_input_unique_cuda_" + std::to_string(task->key) + "_0");
        get_neighbour_time = tt.Passed();
        nbr_cb(task);
        prefetc_improved.Reset();
      }
    } else {
      hash_table->FillWithDupRevised(out_dst, num_samples,
                                    sample_stream);
      hash_table->RefUnique(unique, &num_unique);
    }

    double populate_time = t2.Passed();

    double remap_time = t1.Passed();

    double layer_time = tlayer.Passed();

    auto train_graph = std::make_shared<TrainGraph>();
    train_graph->num_src = num_unique;
    train_graph->num_dst = num_input;
    train_graph->num_edge = num_samples;
    train_graph->col = Tensor::FromBlob(
        out_src, DataType::kI32, {num_samples}, sampler_ctx,
        "train_graph.row_cuda_sample_" + std::to_string(task->key) + "_" +
            std::to_string(i));
    train_graph->row = Tensor::FromBlob(
        out_dst, DataType::kI32, {num_samples}, sampler_ctx,
        "train_graph.dst_cuda_sample_" + std::to_string(task->key) + "_" +
            std::to_string(i));

    task->graphs[i] = train_graph;

    // Do some clean jobs
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
    input = unique;
    num_input = num_unique;
    LOG(DEBUG) << "GPUSample: finish layer " << i;
  }

  // remap edges
  size_t total_num_samples = 0;
  for (int i = last_layer_idx; i>=0; i--) {
    std::shared_ptr<TrainGraph> train_graph = task->graphs[i];
    size_t num_samples = train_graph->num_edge;
    IdType *new_src = sampler_device->AllocArray<IdType>(sampler_ctx, num_samples);
    IdType *new_dst = sampler_device->AllocArray<IdType>(sampler_ctx, num_samples);
    std::shared_ptr<Tensor> col = train_graph->col;
    IdType *old_src = train_graph->col->Ptr<IdType>();
    IdType *old_dst = train_graph->row->Ptr<IdType>();
    GPUMapEdges(old_src, new_src, old_dst, new_dst, num_samples, hash_table->DeviceHandle(), sampler_ctx, sample_stream);
    train_graph->col->ReplaceData(static_cast<void*>(new_src));
    train_graph->row->ReplaceData(static_cast<void*>(new_dst));
    total_num_samples += num_samples;
  }

  task->graph_remapped.store(true, std::memory_order_release);
  double prefetch_improved =  prefetc_improved.Passed();
  LOG(DEBUG) << "edge remapping done " << task->key;

  LOG(DEBUG) << "task_key=" << task->key << " total_num_samples " << total_num_samples;
  Profiler::Get().LogStep(task->key, kLogL1NumNode, num_input);
  Profiler::Get().LogStep(task->key, kLogL1NumSample, total_num_samples);
  Profiler::Get().LogStepAdd(task->key, kLogL3RemapFillUniqueTime,
                             fill_unique_time);
  Profiler::Get().LogStep(task->key, kLogL1PrefetchAdvanced, prefetch_improved);
  Profiler::Get().LogStep(task->key, kLogL1GetNeighbourTime, get_neighbour_time);


  LOG(DEBUG) << "SampleLoop: process task with key " << task->key;
}

void DoGPUSampleAllNeighbour(TaskPtr task) {
  auto num_layers = GPUEngine::Get()->GetFanout().size();
  auto last_layer_idx = num_layers - 1;

  auto dataset = GPUEngine::Get()->GetGraphDataset();
  auto sampler_ctx = GPUEngine::Get()->GetSamplerCtx();
  auto sampler_device = Device::Get(sampler_ctx);
  auto sample_stream = GPUEngine::Get()->GetSampleStream();

  OrderedHashTable *hash_table = GPUEngine::Get()->GetHashtable();
  hash_table->Reset(sample_stream);

  Timer t;
  auto output_nodes = task->output_nodes;
  size_t num_train_node = output_nodes->Shape()[0];
  hash_table->FillWithUnique(output_nodes->CPtr<IdType>(), num_train_node, sample_stream);
  task->graphs.resize(num_layers);

  const IdType *indptr = dataset->indptr->CPtr<IdType>();
  const IdType *indices = dataset->indices->CPtr<IdType>();

  const IdType* input = task->output_nodes->CPtr<IdType>();
  size_t num_input = task->output_nodes->Shape()[0];

  for (int i = last_layer_idx; i >= 0; i--) {
    Timer tlayer;
    Timer t0;
    // const size_t fanout = fanouts[i];
    LOG(DEBUG) << "DoGPUSample: begin sample layer " << i;

    Timer t1;
    Timer t2;

    IdType num_unique;
    const IdType *unique;

    IdType *nbrs;
    size_t num_nbrs_dup;
    GPUExtractNeighbour(indptr, indices, input, num_input, nbrs, &num_nbrs_dup, sampler_ctx, sample_stream, task->key);
    hash_table->FillWithDupMutable(nbrs, num_nbrs_dup, sample_stream);
    sampler_device->FreeWorkspace(sampler_ctx, nbrs);
    hash_table->RefUnique(unique, &num_unique);

    // auto train_graph = std::make_shared<TrainGraph>();
    // train_graph->num_src = num_unique;
    // train_graph->num_dst = num_input;

    // task->graphs[i] = train_graph;

    // Do some clean jobs
    // Profiler::Get().LogStepAdd(task->key, kLogL2CoreSampleTime,
    //                            core_sample_time);
    // Profiler::Get().LogStepAdd(task->key, kLogL2IdRemapTime, remap_time);
    // Profiler::Get().LogStepAdd(task->key, kLogL3RemapPopulateTime,
    //                            populate_time);
    // Profiler::Get().LogStepAdd(task->key, kLogL3RemapMapNodeTime, 0);
    input = unique;
    num_input = num_unique;
    LOG(DEBUG) << "GPUSample: finish layer " << i;
  }

  task->input_nodes = Tensor::CopyBlob(
      input, DataType::kI32, {num_input}, sampler_ctx, sampler_ctx, 
      "cur_input_unique_cuda_" + std::to_string(task->key) + "_0");

  // Profiler::Get().LogStep(task->key, kLogL1NumNode, num_input);
  // Profiler::Get().LogStep(task->key, kLogL1NumSample, total_num_samples);
  // Profiler::Get().LogStepAdd(task->key, kLogL3RemapFillUniqueTime,
  //                            fill_unique_time);
  // Profiler::Get().LogStep(task->key, kLogL1PrefetchAdvanced, prefetch_improved);
  // Profiler::Get().LogStep(task->key, kLogL1GetNeighbourTime, get_neighbour_time);

  LOG(DEBUG) << "SampleLoop: process task with key " << task->key;
}

void DoGraphCopy(TaskPtr task) {
  auto sampler_ctx = GPUEngine::Get()->GetSamplerCtx();
  auto trainer_ctx = GPUEngine::Get()->GetTrainerCtx();
  auto sampler_device = Device::Get(sampler_ctx);
  auto copy_stream = GPUEngine::Get()->GetSamplerCopyStream();

  for (size_t i = 0; i < task->graphs.size(); i++) {
    auto graph = task->graphs[i];
    graph->row = Tensor::CopyTo(graph->row, trainer_ctx, copy_stream,
        "train_graph.row_cuda_train_" + std::to_string(task->key) + "_" + std::to_string(i));
    graph->col = Tensor::CopyTo(graph->col, trainer_ctx, copy_stream,
        "train_graph.col_cuda_train_" + std::to_string(task->key) + "_" + std::to_string(i));

    LOG(DEBUG) << "GraphCopyDevice2DeviceLoop: cuda train_row malloc "
               << ToReadableSize(graph->row->NumBytes());
    LOG(DEBUG) << "GraphCopyDevice2DeviceLoop: cuda train_col malloc "
               << ToReadableSize(graph->col->NumBytes());

    if (RunConfig::sample_type == kRandomWalk) {
      CHECK(graph->data != nullptr);
      graph->data = Tensor::CopyTo(graph->data, trainer_ctx, copy_stream,
        "train_graph.data_cuda_train_" + std::to_string(task->key) + "_" + std::to_string(i));
      LOG(DEBUG) << "GraphCopyDevice2DeviceLoop: cuda graph data malloc "
                 << ToReadableSize(graph->data->NumBytes());
    }

    sampler_device->StreamSync(sampler_ctx, copy_stream);

    Profiler::Get().LogStepAdd(task->key, kLogL1GraphBytes,
                               graph->row->NumBytes() + graph->col->NumBytes());
  }

  LOG(DEBUG) << "GraphCopyDevice2Device: process task with key " << task->key;
}

void DoIdCopy(TaskPtr task) {
  auto sampler_ctx = GPUEngine::Get()->GetSamplerCtx();
  auto sampler_device = Device::Get(sampler_ctx);
  auto copy_stream = GPUEngine::Get()->GetSamplerCopyStream();

  task->input_nodes = Tensor::CopyTo(task->input_nodes, CPU(), copy_stream,
      "task.input_nodes_cpu_" + std::to_string(task->key));
  task->output_nodes = Tensor::CopyTo(task->output_nodes, CPU(), copy_stream,
      "task.output_nodes_cpu_" + std::to_string(task->key));
  LOG(DEBUG) << "IdCopyDevice2Host input_nodes cpu malloc "
             << ToReadableSize(task->input_nodes->NumBytes());
  LOG(DEBUG) << "IdCopyDevice2Host output_nodes cpu malloc "
             << ToReadableSize(task->output_nodes->NumBytes());

  sampler_device->StreamSync(sampler_ctx, copy_stream);

  Profiler::Get().LogStepAdd(
      task->key, kLogL1IdBytes,
      task->input_nodes->NumBytes() + task->output_nodes->NumBytes());

  LOG(DEBUG) << "IdCopyDevice2Host: process task with key " << task->key;
}

void DoCPUFeatureExtract(TaskPtr task) {
  auto dataset = GPUEngine::Get()->GetGraphDataset();

  auto input_nodes = task->input_nodes;
  auto output_nodes = task->output_nodes;

  auto feat = dataset->feat;
  auto label = dataset->label;

  auto feat_dim = feat->Shape()[1];
  auto feat_type = feat->Type();
  auto label_type = dataset->label->Type();

  auto input_data = input_nodes->CPtr<IdType>();
  auto output_data = output_nodes->CPtr<IdType>();
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
  auto feat_src = feat->Data();

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

  if (RunConfig::option_log_node_access || RunConfig::option_log_node_access_simple) {
    Profiler::Get().LogNodeAccess(task->key, input_data, num_input);
  }

  LOG(DEBUG) << "HostFeatureExtract: process task with key " << task->key;
}

void DoGPUFeatureExtract(TaskPtr task) {
  auto dataset = GPUEngine::Get()->GetGraphDataset();
  auto sampler_ctx = GPUEngine::Get()->GetSamplerCtx();
  auto sample_stream = GPUEngine::Get()->GetSampleStream();

  auto input_nodes = task->input_nodes;
  auto output_nodes = task->output_nodes;

  auto feat = dataset->feat;
  auto label = dataset->label;

  auto feat_dim = dataset->feat->Shape()[1];
  auto feat_type = dataset->feat->Type();
  auto label_type = dataset->label->Type();

  auto input_data = input_nodes->CPtr<IdType>();
  auto output_data = output_nodes->CPtr<IdType>();
  auto num_input = input_nodes->Shape()[0];
  auto num_ouput = output_nodes->Shape()[0];

  task->input_feat =
      Tensor::Empty(feat_type, {num_input, feat_dim}, sampler_ctx,
                    "task.input_feat_cuda_" + std::to_string(task->key));
  task->output_label =
      Tensor::Empty(label_type, {num_ouput}, sampler_ctx,
                    "task.output_label_cuda" + std::to_string(task->key));

  auto feat_dst = task->input_feat->MutableData();
  auto feat_src = dataset->feat->Data();

  if (RunConfig::option_empty_feat != 0) {
    GPUMockExtract(feat_dst, feat_src, input_data, num_input, feat_dim,
                   feat_type, sampler_ctx, sample_stream, task->key);
  } else {
    GPUExtract(feat_dst, feat_src, input_data, num_input, feat_dim, feat_type,
              sampler_ctx, sample_stream, task->key);
  }

  auto label_dst = task->output_label->MutableData();
  auto label_src = dataset->label->Data();
  GPUExtract(label_dst, label_src, output_data, num_ouput, 1, label_type,
             sampler_ctx, sample_stream, task->key);

  LOG(DEBUG) << "GPUFeatureExtract: process task with key " << task->key;
}

void DoGPULabelExtract(TaskPtr task) {
  auto dataset = GPUEngine::Get()->GetGraphDataset();

  auto trainer_ctx = GPUEngine::Get()->GetTrainerCtx();
  auto trainer_device = Device::Get(trainer_ctx);
  auto trainer_copy_stream = GPUEngine::Get()->GetTrainerCopyStream();

  auto output_nodes = task->output_nodes;
  auto label = dataset->label;
  auto label_type = dataset->label->Type();

  auto output_data = output_nodes->CPtr<IdType>();
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

  GPUExtract(label_dst, label_src, output_data, num_ouput, 1, label_type,
             trainer_ctx, trainer_copy_stream, task->key);

  LOG(DEBUG) << "HostFeatureExtract: process task with key " << task->key;
  /** SXN: this copy is buggy! */
  // trainer_device->CopyDataFromTo(label_dst, 0, train_label->MutableData(), 0,
  //                                GetTensorBytes(label_type, {num_ouput}), CPU(),
  //                                train_label->Ctx(), trainer_copy_stream);
  trainer_device->StreamSync(trainer_ctx, trainer_copy_stream);

  task->output_label = train_label;

  Profiler::Get().LogStep(task->key, kLogL1LabelBytes, train_label->NumBytes());
}

void DoCPULabelExtractAndCopy(TaskPtr task) {
  auto dataset = GPUEngine::Get()->GetGraphDataset();
  auto trainer_ctx = GPUEngine::Get()->GetTrainerCtx();
  auto trainer_device = Device::Get(trainer_ctx);
  auto copy_stream = GPUEngine::Get()->GetTrainerCopyStream();

  // 1. Extract
  auto output_nodes = task->output_nodes;

  auto label = dataset->label;
  auto label_type = dataset->label->Type();

  auto output_data = output_nodes->CPtr<IdType>();
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
  auto train_label = Tensor::CopyTo(cpu_label, trainer_ctx, copy_stream,
      "task.train_label_cuda" + std::to_string(task->key));
  trainer_device->StreamSync(trainer_ctx, copy_stream);

  task->output_label = train_label;

  Profiler::Get().LogStep(task->key, kLogL1LabelBytes, train_label->NumBytes());

  LOG(DEBUG) << "DoCPULabelExtractAndCopy: process task with key " << task->key;
}

void DoFeatureCopy(TaskPtr task) {
  auto trainer_ctx = GPUEngine::Get()->GetTrainerCtx();
  auto trainer_device = Device::Get(trainer_ctx);
  auto copy_stream = GPUEngine::Get()->GetTrainerCopyStream();

  auto cpu_feat = task->input_feat;
  auto cpu_label = task->output_label;

  CHECK_EQ(cpu_feat->Ctx().device_type, CPU().device_type);
  CHECK_EQ(cpu_label->Ctx().device_type, CPU().device_type);

  auto train_feat = Tensor::CopyTo(cpu_feat, trainer_ctx, copy_stream,
      "task.train_feat_cuda_" + std::to_string(task->key), Constant::kAllocScale);
  auto train_label = Tensor::CopyTo(cpu_label, trainer_ctx, copy_stream,
      "task.train_label_cuda" + std::to_string(task->key), Constant::kAllocScale);
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
  auto sampler_ctx = GPUEngine::Get()->GetSamplerCtx();
  auto trainer_ctx = GPUEngine::Get()->GetTrainerCtx();
  auto sampler_device = Device::Get(sampler_ctx);
  auto copy_stream = GPUEngine::Get()->GetSamplerCopyStream();

  task->output_nodes = Tensor::CopyTo(task->output_nodes, trainer_ctx, copy_stream,
      "task.output_nodes_cuda_" + std::to_string(task->key));
  LOG(DEBUG) << "IdCopyDevice2Host output_nodes cuda malloc "
             << ToReadableSize(task->output_nodes->NumBytes());

  sampler_device->StreamSync(sampler_ctx, copy_stream);

  Profiler::Get().LogStepAdd(task->key, kLogL1IdBytes,
                             task->output_nodes->NumBytes());

  LOG(DEBUG) << "IdCopyDevice2Host: process task with key " << task->key;
}

void DoCacheIdCopyToCPU(TaskPtr task) {
  auto sampler_ctx = GPUEngine::Get()->GetSamplerCtx();
  auto sampler_device = Device::Get(sampler_ctx);
  auto copy_stream = GPUEngine::Get()->GetSamplerCopyStream();

  task->output_nodes = Tensor::CopyTo(task->output_nodes, CPU(), copy_stream,
      "task.output_nodes_cuda_" + std::to_string(task->key));
  LOG(DEBUG) << "IdCopyDevice2Host output_nodes cuda malloc "
             << ToReadableSize(task->output_nodes->NumBytes());

  sampler_device->StreamSync(sampler_ctx, copy_stream);

  Profiler::Get().LogStepAdd(task->key, kLogL1IdBytes,
                             task->output_nodes->NumBytes());

  LOG(DEBUG) << "IdCopyDevice2Host: process task with key " << task->key;
}

void DoCacheFeatureCopy(TaskPtr task) {
  auto trainer_ctx = GPUEngine::Get()->GetTrainerCtx();
  auto trainer_device = Device::Get(trainer_ctx);
  auto sampler_ctx = GPUEngine::Get()->GetSamplerCtx();
  auto sampler_device = Device::Get(sampler_ctx);
  auto cpu_device = Device::Get(CPU());
  auto sampler_copy_stream = GPUEngine::Get()->GetSamplerCopyStream();
  auto trainer_copy_stream = GPUEngine::Get()->GetTrainerCopyStream();

  auto dataset = GPUEngine::Get()->GetGraphDataset();
  auto cache_manager = GPUEngine::Get()->GetCacheManager();

  auto input_nodes = task->input_nodes;
  auto feat = dataset->feat;
  auto feat_dim = dataset->feat->Shape()[1];
  auto feat_type = dataset->feat->Type();

  auto input_data = input_nodes->CPtr<IdType>();
  auto num_input = input_nodes->Shape()[0];

  CHECK_EQ(input_nodes->Ctx().device_type, sampler_ctx.device_type);
  CHECK_EQ(input_nodes->Ctx().device_id, sampler_ctx.device_id);

  auto train_feat =
      Tensor::Empty(feat_type, {num_input, feat_dim}, trainer_ctx,
                    "task.train_feat_cuda_" + std::to_string(task->key));

  // 0. Get index of miss data and cache data
  // feature data has cache, so we only need to extract the miss data
  Timer t0;

  IdType *sampler_output_miss_src_index =
      sampler_device->AllocArray<IdType>(sampler_ctx, num_input);
  IdType *sampler_output_miss_dst_index =
      sampler_device->AllocArray<IdType>(sampler_ctx, num_input);
  IdType *sampler_output_cache_src_index =
      sampler_device->AllocArray<IdType>(sampler_ctx, num_input);
  IdType *sampler_output_cache_dst_index =
      sampler_device->AllocArray<IdType>(sampler_ctx, num_input);

  size_t num_output_miss;
  size_t num_output_cache;

  cache_manager->GetMissCacheIndex(
      sampler_output_miss_src_index, sampler_output_miss_dst_index,
      &num_output_miss, sampler_output_cache_src_index,
      sampler_output_cache_dst_index, &num_output_cache, input_data, num_input,
      sampler_copy_stream);

  CHECK_EQ(num_output_miss + num_output_cache, num_input);

  double get_index_time = t0.Passed();

  // 1. Move the miss index

  Timer t1;

  IdType *cpu_output_miss_src_index = 
      cpu_device->AllocArray<IdType>(CPU(), num_output_miss);
  IdType *trainer_output_miss_dst_index =
      trainer_device->AllocArray<IdType>(trainer_ctx, num_output_miss);
  IdType *trainer_output_cache_src_index =
      trainer_device->AllocArray<IdType>(trainer_ctx, num_output_cache);
  IdType *trainer_output_cache_dst_index =
      trainer_device->AllocArray<IdType>(trainer_ctx, num_output_cache);

  sampler_device->CopyDataFromTo(sampler_output_miss_src_index, 0,
                                 cpu_output_miss_src_index, 0,
                                 num_output_miss * sizeof(IdType), sampler_ctx,
                                 CPU(), sampler_copy_stream);
  sampler_device->CopyDataFromTo(sampler_output_miss_dst_index, 0,
                                 trainer_output_miss_dst_index, 0,
                                 num_output_miss * sizeof(IdType), sampler_ctx,
                                 trainer_ctx, sampler_copy_stream);
  trainer_device->CopyDataFromTo(sampler_output_cache_src_index, 0,
                                 trainer_output_cache_src_index, 0,
                                 num_output_cache * sizeof(IdType), sampler_ctx,
                                 trainer_ctx, trainer_copy_stream);
  trainer_device->CopyDataFromTo(sampler_output_cache_dst_index, 0,
                                 trainer_output_cache_dst_index, 0,
                                 num_output_cache * sizeof(IdType), sampler_ctx,
                                 trainer_ctx, trainer_copy_stream);

  sampler_device->StreamSync(sampler_ctx, sampler_copy_stream);
  trainer_device->StreamSync(trainer_ctx, trainer_copy_stream);

  sampler_device->FreeWorkspace(sampler_ctx, sampler_output_miss_src_index);
  sampler_device->FreeWorkspace(sampler_ctx, sampler_output_miss_dst_index);
  sampler_device->FreeWorkspace(sampler_ctx, sampler_output_cache_src_index);
  sampler_device->FreeWorkspace(sampler_ctx, sampler_output_cache_dst_index);

  double copy_idx_time = t1.Passed();

  // 2. Extract the miss data
  Timer t2;
  auto cpu_output_miss = Tensor::Empty(
      feat_type, {num_output_miss, feat_dim}, CPU(), "");

  cache_manager->ExtractMissData(cpu_output_miss->MutableData(),
                                 cpu_output_miss_src_index, num_output_miss);

  double extract_miss_time = t2.Passed();

  // 3. Copy the miss data
  Timer t3;
  auto trainer_output_miss = Tensor::CopyTo(
      cpu_output_miss, trainer_ctx, trainer_copy_stream);
  trainer_device->StreamSync(trainer_ctx, trainer_copy_stream);

  cpu_output_miss = nullptr;

  double copy_miss_time = t3.Passed();

  // 4. Combine miss data
  Timer t4;
  cache_manager->CombineMissData(train_feat->MutableData(), trainer_output_miss->Data(),
                                 trainer_output_miss_dst_index, num_output_miss,
                                 trainer_copy_stream);
  trainer_device->StreamSync(trainer_ctx, trainer_copy_stream);

  double combine_miss_time = t4.Passed();

  // 5. Combine cache data
  Timer t5;
  cache_manager->CombineCacheData(
      train_feat->MutableData(), trainer_output_cache_src_index,
      trainer_output_cache_dst_index, num_output_cache, trainer_copy_stream);
  trainer_device->StreamSync(trainer_ctx, trainer_copy_stream);

  double combine_cache_time = t5.Passed();

  task->input_feat = train_feat;

  // 5. Free space
  cpu_device->FreeWorkspace(CPU(), cpu_output_miss_src_index);
  trainer_output_miss = nullptr;
  trainer_device->FreeWorkspace(trainer_ctx, trainer_output_miss_dst_index);
  trainer_device->FreeWorkspace(trainer_ctx, trainer_output_cache_src_index);
  trainer_device->FreeWorkspace(trainer_ctx, trainer_output_cache_dst_index);

  Profiler::Get().LogStep(task->key, kLogL1FeatureBytes,
                          train_feat->NumBytes());
  Profiler::Get().LogStep(
      task->key, kLogL1MissBytes,
      GetTensorBytes(feat_type, {num_output_miss, feat_dim}));
  Profiler::Get().LogStep(task->key, kLogL3CacheGetIndexTime, get_index_time); // t0, step 0
  Profiler::Get().LogStep(task->key, KLogL3CacheCopyIndexTime, copy_idx_time); // t1, step 1
  Profiler::Get().LogStep(task->key, kLogL3CacheExtractMissTime,
                          extract_miss_time); // t2, step2
  Profiler::Get().LogStep(task->key, kLogL3CacheCopyMissTime, copy_miss_time); // t3, step 3
  Profiler::Get().LogStep(task->key, kLogL3CacheCombineMissTime,
                          combine_miss_time);  // t4, step 4
  Profiler::Get().LogStep(task->key, kLogL3CacheCombineCacheTime,
                          combine_cache_time); // t5, step 5
  Profiler::Get().LogEpochAdd(task->key, kLogEpochFeatureBytes,
                              train_feat->NumBytes());
  Profiler::Get().LogEpochAdd(
      task->key, kLogEpochMissBytes,
      GetTensorBytes(feat_type, {num_output_miss, feat_dim}));

  LOG(DEBUG) << "DoCacheFeatureCopy: process task with key " << task->key;
}
/**
 * @brief FIXME: remove redundant code
 */
void DoDynamicCacheFeatureCopy(TaskPtr task) {
  auto trainer_ctx = GPUEngine::Get()->GetTrainerCtx();
  auto trainer_device = Device::Get(trainer_ctx);
  auto sampler_ctx = GPUEngine::Get()->GetSamplerCtx();
  auto sampler_device = Device::Get(sampler_ctx);
  auto cpu_device = Device::Get(CPU());
  auto sampler_copy_stream = GPUEngine::Get()->GetSamplerCopyStream();
  auto trainer_copy_stream = GPUEngine::Get()->GetTrainerCopyStream();

  auto dataset = GPUEngine::Get()->GetGraphDataset();
  auto cache_manager = GPUEngine::Get()->GetDynamicCacheManager();

  auto input_nodes = task->input_nodes;
  auto feat = dataset->feat;
  auto feat_dim = dataset->feat->Shape()[1];
  auto feat_type = dataset->feat->Type();

  auto input_data = input_nodes->CPtr<IdType>();
  auto num_input = input_nodes->Shape()[0];

  CHECK_EQ(input_nodes->Ctx().device_type, sampler_ctx.device_type);
  CHECK_EQ(input_nodes->Ctx().device_id, sampler_ctx.device_id);

  auto train_feat =
      Tensor::Empty(feat_type, {num_input, feat_dim}, trainer_ctx,
                    "task.train_feat_cuda_" + std::to_string(task->key));

  // 0. Get index of miss data and cache data
  // feature data has cache, so we only need to extract the miss data
  Timer t0;

  IdType *sampler_output_miss_src_index =
      sampler_device->AllocArray<IdType>(sampler_ctx, num_input);
  IdType *sampler_output_miss_dst_index =
      sampler_device->AllocArray<IdType>(sampler_ctx, num_input);
  IdType *sampler_output_cache_src_index =
      sampler_device->AllocArray<IdType>(sampler_ctx, num_input);
  IdType *sampler_output_cache_dst_index =
      sampler_device->AllocArray<IdType>(sampler_ctx, num_input);

  size_t num_output_miss;
  size_t num_output_cache;

  cache_manager->GetMissCacheIndex(
      sampler_output_miss_src_index, sampler_output_miss_dst_index,
      &num_output_miss, sampler_output_cache_src_index,
      sampler_output_cache_dst_index, &num_output_cache, input_data, num_input,
      sampler_copy_stream);

  CHECK_EQ(num_output_miss + num_output_cache, num_input);

  double get_index_time = t0.Passed();

  // 1. Move the miss index

  Timer t1;

  IdType *cpu_output_miss_src_index =
      cpu_device->AllocArray<IdType>(CPU(), num_output_miss);
  IdType *trainer_output_miss_dst_index =
      trainer_device->AllocArray<IdType>(trainer_ctx, num_output_miss);
  IdType *trainer_output_cache_src_index =
      trainer_device->AllocArray<IdType>(trainer_ctx, num_output_cache);
  IdType *trainer_output_cache_dst_index =
      trainer_device->AllocArray<IdType>(trainer_ctx, num_output_cache);

  sampler_device->CopyDataFromTo(sampler_output_miss_src_index, 0,
                                 cpu_output_miss_src_index, 0,
                                 num_output_miss * sizeof(IdType), sampler_ctx,
                                 CPU(), sampler_copy_stream);
  sampler_device->CopyDataFromTo(sampler_output_miss_dst_index, 0,
                                 trainer_output_miss_dst_index, 0,
                                 num_output_miss * sizeof(IdType), sampler_ctx,
                                 trainer_ctx, sampler_copy_stream);
  trainer_device->CopyDataFromTo(sampler_output_cache_src_index, 0,
                                 trainer_output_cache_src_index, 0,
                                 num_output_cache * sizeof(IdType), sampler_ctx,
                                 trainer_ctx, trainer_copy_stream);
  trainer_device->CopyDataFromTo(sampler_output_cache_dst_index, 0,
                                 trainer_output_cache_dst_index, 0,
                                 num_output_cache * sizeof(IdType), sampler_ctx,
                                 trainer_ctx, trainer_copy_stream);

  sampler_device->StreamSync(sampler_ctx, sampler_copy_stream);
  trainer_device->StreamSync(trainer_ctx, trainer_copy_stream);

  sampler_device->FreeWorkspace(sampler_ctx, sampler_output_miss_src_index);
  sampler_device->FreeWorkspace(sampler_ctx, sampler_output_miss_dst_index);
  sampler_device->FreeWorkspace(sampler_ctx, sampler_output_cache_src_index);
  sampler_device->FreeWorkspace(sampler_ctx, sampler_output_cache_dst_index);

  double copy_idx_time = t1.Passed();

  // 2. Extract the miss data
  Timer t2;

  auto cpu_output_miss = Tensor::Empty(feat_type, {num_output_miss, feat_dim}, CPU(), ""); 
  cache_manager->ExtractMissData(cpu_output_miss->MutableData(),
                                 cpu_output_miss_src_index, num_output_miss);

  double extract_miss_time = t2.Passed();

  // 3. Copy the miss data
  Timer t3;
  auto trainer_output_miss = Tensor::CopyTo(
      cpu_output_miss, trainer_ctx, trainer_copy_stream);
  trainer_device->StreamSync(trainer_ctx, trainer_copy_stream);

  cpu_output_miss = nullptr;

  double copy_miss_time = t3.Passed();

  // 4. Combine miss data
  Timer t4;
  cache_manager->CombineMissData(train_feat->MutableData(), trainer_output_miss->Data(),
                                 trainer_output_miss_dst_index, num_output_miss,
                                 trainer_copy_stream);
  trainer_device->StreamSync(trainer_ctx, trainer_copy_stream);

  double combine_miss_time = t4.Passed();

  // 5. Combine cache data
  Timer t5;
  cache_manager->CombineCacheData(
      train_feat->MutableData(), trainer_output_cache_src_index,
      trainer_output_cache_dst_index, num_output_cache, trainer_copy_stream);
  trainer_device->StreamSync(trainer_ctx, trainer_copy_stream);

  double combine_cache_time = t5.Passed();

  // 6. Replace dynamic cache
  Timer t6;
  cache_manager->ReplaceCacheGPU(task->input_nodes, train_feat, sampler_copy_stream);
  sampler_device->StreamSync(sampler_ctx, sampler_copy_stream);
  double replace_cache_time = t6.Passed();

  task->input_feat = train_feat;

  // 7. Free space
  cpu_device->FreeWorkspace(CPU(), cpu_output_miss_src_index);
  trainer_output_miss = nullptr;
  trainer_device->FreeWorkspace(trainer_ctx, trainer_output_miss_dst_index);
  trainer_device->FreeWorkspace(trainer_ctx, trainer_output_cache_src_index);
  trainer_device->FreeWorkspace(trainer_ctx, trainer_output_cache_dst_index);

  Profiler::Get().LogStep(task->key, kLogL1FeatureBytes,
                          train_feat->NumBytes());
  Profiler::Get().LogStep(
      task->key, kLogL1MissBytes,
      GetTensorBytes(feat_type, {num_output_miss, feat_dim}));
  Profiler::Get().LogStep(task->key, kLogL3CacheGetIndexTime, get_index_time); // t0, step 0
  Profiler::Get().LogStep(task->key, KLogL3CacheCopyIndexTime, copy_idx_time); // t1, step 1
  Profiler::Get().LogStep(task->key, kLogL3CacheExtractMissTime,
                          extract_miss_time); // t2, step2
  Profiler::Get().LogStep(task->key, kLogL3CacheCopyMissTime, copy_miss_time); // t3, step 3
  Profiler::Get().LogStep(task->key, kLogL3CacheCombineMissTime,
                          combine_miss_time);  // t4, step 4
  Profiler::Get().LogStep(task->key, kLogL3CacheCombineCacheTime,
                          combine_cache_time); // t5, step 5

  LOG(DEBUG) << "DoCacheFeatureCopy: process task with key " << task->key;
}

}  // namespace cuda
}  // namespace common
}  // namespace samgraph