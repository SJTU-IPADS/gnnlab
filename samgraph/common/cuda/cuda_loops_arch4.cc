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

#include <atomic>
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
#include "cuda_loops.h"

/* clang-format off
 * +-----------------------+     +--------------------+     +------------------------+
 * |                       |     |                    |     |                        |
 * |                       |     |                    |     |                        |
 * |                       |     |                    |     |                        |
 * |       Sampling        ------> Feature Extraction ------>        Training        |
 * |                       |     |                    |     |                        |
 * |                       |     |                    |     |                        |
 * | Dedicated Sampler GPU |     |         CPU        |     | Dedicated Trainer GPU  |
 * +-----------------------+     +--------------------+     +------------------------+
 * clang-format on
 */

namespace samgraph {
namespace common {
namespace cuda {

namespace {

bool RunSampleSubLoopOnce() {
  auto next_op = kDataCopy;
  auto next_q = GPUEngine::Get()->GetTaskQueue(next_op);
  while (next_q->Full()) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    // return true;
  }

  Timer t0;
  auto task = DoShuffle();
  if (task) {
    double shuffle_time = t0.Passed();

    std::function <void(TaskPtr)> nbr_cb = [next_q](TaskPtr tp) {
      next_q->AddTask(tp);
    };
    Timer t1;
    DoGPUSampleDyCache(task, nbr_cb);
    double sample_time = t1.Passed();
    Timer t2;
    Profiler::Get().TraceStepBegin(task->key, kL1Event_Sample, t0.TimePointMicro());
    Profiler::Get().TraceStepEnd(task->key, kL1Event_Sample, t2.TimePointMicro());

    Profiler::Get().LogStep(task->key, kLogL1SampleTime,
                            shuffle_time + sample_time);
    Profiler::Get().LogStep(task->key, kLogL2ShuffleTime, shuffle_time);
    Profiler::Get().LogEpochAdd(task->key, kLogEpochSampleTime,
                                shuffle_time + sample_time);
  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }

  return true;
}

bool RunDataCopySubLoopOnce() {
  auto graph_pool = GPUEngine::Get()->GetGraphPool();
  while (graph_pool->Full()) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    // return true;
  }

  auto this_op = kDataCopy;
  auto q = GPUEngine::Get()->GetTaskQueue(this_op);
  auto task = q->GetTask();

  if (task) {
    Timer t2;
    DoIdCopy(task);
    double id_copy_time = t2.Passed();

    Timer t0;
    DoCPUFeatureExtract(task);
    double extract_time = t0.Passed();

    Timer t1;
    DoFeatureCopy(task);
    double feat_copy_time = t1.Passed();


    LOG(DEBUG) << "Waiting for edge remapping " << task->key;
    while(task->graph_remapped.load(std::memory_order_acquire) == false) {
      std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    }
    LOG(DEBUG) << "Copy waited for edge remapping done " << task->key;

    Timer t3;
    DoGraphCopy(task);
    double graph_copy_time = t3.Passed();
    Timer t4;

    LOG(DEBUG) << "Submit: process task with key " << task->key;
    graph_pool->Submit(task->key, task);
    Profiler::Get().TraceStepBegin(task->key, kL1Event_Copy, t2.TimePointMicro());
    Profiler::Get().TraceStepEnd(task->key, kL1Event_Copy, t4.TimePointMicro());
    Profiler::Get().LogStep(
        task->key, kLogL1CopyTime,
        graph_copy_time + id_copy_time + extract_time + feat_copy_time);
    Profiler::Get().LogStep(task->key, kLogL2GraphCopyTime, graph_copy_time);
    Profiler::Get().LogStep(task->key, kLogL2IdCopyTime, id_copy_time);
    Profiler::Get().LogStep(task->key, kLogL2ExtractTime, extract_time);
    Profiler::Get().LogStep(task->key, kLogL2FeatCopyTime, feat_copy_time);
    Profiler::Get().LogEpochAdd(
        task->key, kLogEpochCopyTime,
        graph_copy_time + id_copy_time + extract_time + feat_copy_time);
  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }

  return true;
}

bool RunCacheDataCopySubLoopOnce() {
  auto graph_pool = GPUEngine::Get()->GetGraphPool();
  while (graph_pool->Full()) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    // return true;
  }

  auto this_op = kDataCopy;
  auto q = GPUEngine::Get()->GetTaskQueue(this_op);
  auto task = q->GetTask();

  if (task) {
    Timer t1;
    DoCacheIdCopy(task);
    double id_copy_time = t1.Passed();

    Timer t2;
    DoDynamicCacheFeatureCopy(task);
    DoGPULabelExtract(task);
    double cache_feat_copy_time = t2.Passed();

    LOG(DEBUG) << "Waiting for edge remapping " << task->key;
    while(task->graph_remapped.load(std::memory_order_acquire) == false) {
      std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    }
    LOG(DEBUG) << "Copy waited for edge remapping done " << task->key;

    Timer t0;
    DoGraphCopy(task);
    double graph_copy_time = t0.Passed();
    Timer t4;

    LOG(DEBUG) << "Submit with cache: process task with key " << task->key;
    graph_pool->Submit(task->key, task);

    Profiler::Get().TraceStepBegin(task->key, kL1Event_Copy, t1.TimePointMicro());
    Profiler::Get().TraceStepEnd(task->key, kL1Event_Copy, t4.TimePointMicro());
    Profiler::Get().LogStep(
        task->key, kLogL1CopyTime,
        graph_copy_time + id_copy_time + cache_feat_copy_time);
    Profiler::Get().LogStep(task->key, kLogL2GraphCopyTime, graph_copy_time);
    Profiler::Get().LogStep(task->key, kLogL2IdCopyTime, id_copy_time);
    Profiler::Get().LogStep(task->key, kLogL2CacheCopyTime,
                            cache_feat_copy_time);
    Profiler::Get().LogEpochAdd(
        task->key, kLogEpochCopyTime,
        graph_copy_time + id_copy_time + cache_feat_copy_time);
  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }
  return true;
}

void SampleSubLoop() {
  for (size_t cur_epoch = 0; cur_epoch < GPUEngine::Get()->NumEpoch(); cur_epoch++) {
    if (RunConfig::barriered_epoch == -1 || RunConfig::barriered_epoch == static_cast<int>(cur_epoch)) {
      Engine::Get()->WaitBarrier();
    }
    for (size_t cur_step = 0; cur_step < GPUEngine::Get()->NumStep(); cur_step++) {
      RunSampleSubLoopOnce();
    }
    Engine::Get()->ForwardInnerBarrier();
  }
  while(!GPUEngine::Get()->ShouldShutdown()) {}

  GPUEngine::Get()->ReportThreadFinish();
}

void DataCopySubLoop() {
  LoopOnceFunction func;
  if (RunConfig::UseDynamicGPUCache()) {
    func = RunCacheDataCopySubLoopOnce;
  } else {
    func = RunDataCopySubLoopOnce;
  }

  while (func() && !GPUEngine::Get()->ShouldShutdown()) {
  }

  GPUEngine::Get()->ReportThreadFinish();
}

}  // namespace

void RunArch4LoopsOnce() {
  RunSampleSubLoopOnce();
  if (RunConfig::UseDynamicGPUCache()) {
    RunCacheDataCopySubLoopOnce();
  } else {
    RunDataCopySubLoopOnce();
  }
}

std::vector<LoopFunction> GetArch4Loops() {
  std::vector<LoopFunction> func;

  func.push_back(SampleSubLoop);
  func.push_back(DataCopySubLoop);

  return func;
}

}  // namespace cuda
}  // namespace common
}  // namespace samgraph
