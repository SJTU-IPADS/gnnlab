#include "../profiler.h"
#include "../common.h"
#include "../constant.h"
#include "../logging.h"
#include "../device.h"
#include "dist_loops.h"
#include "dist_engine.h"
#include "pre_sampler.h"
#include <cstring>
#ifdef __linux__
#include <parallel/algorithm>
#else
#include <algorithm>
#endif
#include "../timer.h"

namespace samgraph {
namespace common {
namespace dist {

PreSampler* PreSampler::singleton = nullptr;
PreSampler::PreSampler(TensorPtr input, size_t batch_size, size_t num_nodes) {
  Timer t_init;
  _shuffler = new cuda::GPUShuffler(input, RunConfig::presample_epoch,
                          batch_size, false);
  _num_nodes = num_nodes;
  _num_step = _shuffler->NumStep();
  freq_table = static_cast<Id64Type*>(Device::Get(CPU())->AllocDataSpace(CPU(), sizeof(Id64Type)*_num_nodes));
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for (size_t i = 0; i < _num_nodes; i++) {
    auto nid_ptr = reinterpret_cast<IdType*>(&freq_table[i]);
    *nid_ptr = i;
    *(nid_ptr + 1) = 0;
  }
  Profiler::Get().LogInit(kLogInitL3PresampleInit, t_init.Passed());
}

PreSampler::~PreSampler() {
  Device::Get(CPU())->FreeDataSpace(CPU(), freq_table);
  delete _shuffler;
}

TaskPtr PreSampler::DoPreSampleShuffle() {
  auto s = _shuffler;
  auto batch = s->GetBatch();

  if (batch) {
    auto task = std::make_shared<Task>();
    task->output_nodes = batch;
    task->key = DistEngine::Get()->GetBatchKey(s->Epoch(), s->Step());
    LOG(DEBUG) << "DoShuffle: process task with key " << task->key;
    return task;
  } else {
    return nullptr;
  }
}

void PreSampler::DoPreSample(){
  auto sampler_ctx = DistEngine::Get()->GetSamplerCtx();
  auto sampler_device = Device::Get(sampler_ctx);
  auto cpu_device = Device::Get(CPU());
  for (int e = 0; e < RunConfig::presample_epoch; e++) {
    for (size_t i = 0; i < _num_step; i++) {
      Timer t0;
      auto task = DoPreSampleShuffle();
      switch (RunConfig::cache_policy) {
        case kCacheByPreSample:
          DoGPUSample(task);
          break;
        case kCacheByPreSampleStatic:
          LOG(FATAL) << "kCacheByPreSampleStatic is not implemented in DistEngine now!";
          /*
          DoGPUSampleAllNeighbour(task);
          break;
          */
        default:
          CHECK(0);
      }
      double sample_time = t0.Passed();
      size_t num_inputs = task->input_nodes->Shape()[0];
      Timer t1;
      IdType* input_nodes = static_cast<IdType*>(cpu_device->AllocWorkspace(CPU(), sizeof(IdType)*num_inputs));
      sampler_device->CopyDataFromTo(
        task->input_nodes->Data(), 0, input_nodes, 0,
        num_inputs * sizeof(IdType), task->input_nodes->Ctx(), CPU());
      double copy_time = t1.Passed();
      Timer t2;
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
      for (size_t i = 0; i < num_inputs; i++) {
        auto freq_ptr = reinterpret_cast<IdType*>(&freq_table[input_nodes[i]]);
        *(freq_ptr+1) += 1;
      }
      cpu_device->FreeWorkspace(CPU(), input_nodes);
      double count_time = t2.Passed();
      Profiler::Get().LogInitAdd(kLogInitL3PresampleSample, sample_time);
      Profiler::Get().LogInitAdd(kLogInitL3PresampleCopy, copy_time);
      Profiler::Get().LogInitAdd(kLogInitL3PresampleCount, count_time);
    }
  }
  Timer ts;
#ifdef __linux__
  __gnu_parallel::sort(freq_table, &freq_table[_num_nodes],
                       std::greater<Id64Type>());
#else
  std::sort(freq_table, &freq_table[_num_nodes],
            std::greater<Id64Type>());
#endif
  double sort_time = ts.Passed();
  Profiler::Get().LogInit(kLogInitL3PresampleSort, sort_time);
  Timer t_reset;
  Profiler::Get().ResetStepEpoch();
  Profiler::Get().LogInit(kLogInitL3PresampleReset, t_reset.Passed());
}

TensorPtr PreSampler::GetFreq() {
  auto ranking_freq = Tensor::Empty(DataType::kI32, {_num_nodes}, CPU(), "");
  auto ranking_freq_ptr = static_cast<IdType*>(ranking_freq->MutableData());
  for (size_t i = 0; i < _num_nodes; i++) {
    auto nid_ptr = reinterpret_cast<IdType*>(&freq_table[i]);
    ranking_freq_ptr[i] = *(nid_ptr + 1);
  }
  return ranking_freq;
}
TensorPtr PreSampler::GetRankNode() {
  auto ranking_nodes = Tensor::Empty(DataType::kI32, {_num_nodes}, CPU(), "");
  auto ranking_nodes_ptr = static_cast<IdType*>(ranking_nodes->MutableData());
  Timer t_prepare_rank;
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for (size_t i = 0; i < _num_nodes; i++) {
    auto nid_ptr = reinterpret_cast<IdType*>(&freq_table[i]);
    ranking_nodes_ptr[i] = *(nid_ptr);
  }
  Profiler::Get().LogInit(kLogInitL3PresampleGetRank, t_prepare_rank.Passed());
  return ranking_nodes;
}
void PreSampler::GetRankNode(TensorPtr& ranking_nodes) {
  auto ranking_nodes_ptr = static_cast<IdType*>(ranking_nodes->MutableData());
  Timer t_prepare_rank;
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for (size_t i = 0; i < _num_nodes; i++) {
    auto nid_ptr = reinterpret_cast<IdType*>(&freq_table[i]);
    ranking_nodes_ptr[i] = *(nid_ptr);
  }
  Profiler::Get().LogInit(kLogInitL3PresampleGetRank, t_prepare_rank.Passed());
}

}
}
}
