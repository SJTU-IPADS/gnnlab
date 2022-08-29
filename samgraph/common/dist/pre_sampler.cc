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
  LOG(ERROR) << "Dist Presampler making shuffler...";
  _shuffler = new cuda::GPUAlignedShuffler(input, RunConfig::presample_epoch,
                          batch_size, false);
  LOG(ERROR) << "Dist Presampler making shuffler...Done";
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
  auto stream = DistEngine::Get()->GetSampleStream();
  auto batch = s->GetBatch(stream);

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
  auto sampler_stream = DistEngine::Get()->GetSamplerCopyStream();
  auto cpu_device = Device::Get(CPU());
  size_t max_num_inputs = 0, min_num_inputs = std::numeric_limits<size_t>::max();
  for (int e = 0; e < RunConfig::presample_epoch; e++) {
    LOG(ERROR) << "Dist Presampler doing presample epoch " << e;
    for (size_t i = 0; i < _num_step; i++) {
      Timer t0;
      auto task = DoPreSampleShuffle();
      switch (RunConfig::cache_policy) {
        case kCollCacheAsymmLink:
        case kCollCacheIntuitive:
        case kCollCache:
        case kPartRepCache:
        case kRepCache:
        case kPartitionCache:
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
      max_num_inputs = std::max(num_inputs, max_num_inputs);
      min_num_inputs = std::min(num_inputs, min_num_inputs);
      Timer t1;
      auto input_nodes = Tensor::CopyTo(task->input_nodes, CPU(), sampler_stream);
      sampler_device->StreamSync(sampler_ctx, sampler_stream);
      double copy_time = t1.Passed();
      Timer t2;
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
      for (size_t i = 0; i < num_inputs; i++) {
        auto freq_ptr = reinterpret_cast<IdType*>(&freq_table[input_nodes->CPtr<IdType>()[i]]);
        *(freq_ptr+1) += 1;
      }
      double count_time = t2.Passed();
      Profiler::Get().LogInitAdd(kLogInitL3PresampleSample, sample_time);
      Profiler::Get().LogInitAdd(kLogInitL3PresampleCopy, copy_time);
      Profiler::Get().LogInitAdd(kLogInitL3PresampleCount, count_time);
    }
    LOG(ERROR) << "presample spend "
               << Profiler::Get().GetLogInitValue(kLogInitL3PresampleSample) << " on sample, "
               << Profiler::Get().GetLogInitValue(kLogInitL3PresampleCopy) << " on copy, "
               << Profiler::Get().GetLogInitValue(kLogInitL3PresampleCount) << " on count";
  }
  double & sf = DistEngine::Get()->GetGraphDataset()->scale_factor->Ptr<double>()[0];
  sf = std::max(sf, max_num_inputs / (double)min_num_inputs + 0.03);
  LOG(ERROR) << "max_num_inputs = " << max_num_inputs << ", min_num_inputs" << min_num_inputs;
  std::cout << "test_result:init:input_scale_factor=" << sf << "\n";
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
  LOG(ERROR) << "presample spend " << sort_time << " on sort freq.";
  Timer t_reset;
  Profiler::Get().ResetStepEpoch();
  Profiler::Get().LogInit(kLogInitL3PresampleReset, t_reset.Passed());
}

TensorPtr PreSampler::GetFreq() {
  auto ranking_freq = Tensor::Empty(DataType::kI32, {_num_nodes}, CPU(), "");
  auto ranking_freq_ptr = static_cast<IdType*>(ranking_freq->MutableData());
  GetFreq(ranking_freq_ptr);
  return ranking_freq;
}

void PreSampler::GetFreq(IdType* ranking_freq_ptr) {
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for (size_t i = 0; i < _num_nodes; i++) {
    auto nid_ptr = reinterpret_cast<IdType*>(&freq_table[i]);
    ranking_freq_ptr[i] = *(nid_ptr + 1);
  }
}
TensorPtr PreSampler::GetRankNode() {
  auto ranking_nodes = Tensor::Empty(DataType::kI32, {_num_nodes}, CPU(), "");
  GetRankNode(ranking_nodes);
  return ranking_nodes;
}
void PreSampler::GetRankNode(TensorPtr& ranking_nodes) {
  auto ranking_nodes_ptr = ranking_nodes->Ptr<IdType>();
  GetRankNode(ranking_nodes_ptr);
}

void PreSampler::GetRankNode(IdType* ranking_nodes_ptr) {
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
