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

  auto cur_input = task->output_nodes;

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
  Profiler::Get().LogStepAdd(task->key, kLogL3RemapFillUniqueTime,
                             fill_unique_time);

  LOG(DEBUG) << "SampleLoop: process task with key " << task->key;
}

void DoGraphCopy(TaskPtr task) {
  auto trainer_ctx = DistEngine::Get()->GetTrainerCtx();
  auto copy_ctx = trainer_ctx;
  auto copy_device = Device::Get(trainer_ctx);
  auto copy_stream = DistEngine::Get()->GetTrainerCopyStream();

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

    copy_device->CopyDataFromTo(graph->row->Data(), 0,
                                   train_row->MutableData(), 0,
                                   graph->row->NumBytes(), graph->row->Ctx(),
                                   train_row->Ctx(), copy_stream);
    copy_device->CopyDataFromTo(graph->col->Data(), 0,
                                   train_col->MutableData(), 0,
                                   graph->col->NumBytes(), graph->col->Ctx(),
                                   train_col->Ctx(), copy_stream);

    // XXX: Random walk is successful here?
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
  auto copy_ctx = CPU();
  auto copy_device = Device::Get(copy_ctx);
  auto copy_stream = nullptr;

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

  LOG(DEBUG) << "FeatureCopyHost2Device: process task with key " << task->key;
}

} // dist
} // common
} // samgraph
