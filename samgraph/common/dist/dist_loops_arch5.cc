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

#include "../cuda/cuda_loops.h"

#include <chrono>
#include <numeric>

#include "../device.h"
#include "../function.h"
#include "../logging.h"
#include "../profiler.h"
#include "../run_config.h"
#include "../timer.h"

#include "dist_loops.h"
#include "../cuda/cuda_function.h"
#include "../cuda/cuda_hashtable.h"
#include "../cuda/cuda_loops.h"

/* clang-format off
 * +-----------------------+        +--------------------+     +------------------------+
 * |                       |        |                    |     |                        |
 * |       Sampling        --Queue--> Feature Extraction ------>        Training        |
 * |                       |     |  |                    |     |                        |
 * | Dedicated Sampler GPU |     |  |         CPU        |     | Dedicated Trainer GPU  |
 * +-----------------------+     |  +--------------------+     +------------------------+
 *                               |
 *                               |  +--------------------+     +------------------------+
 *                               |  |                    |     |                        |
 *                               \--> Feature Extraction ------>        Training        |
 *                                  |                    |     |                        |
 *                                  |         CPU        |     | Dedicated Trainer GPU  |
 *                                  +--------------------+     +------------------------+
 * clang-format on
 */

namespace samgraph {
namespace common {
namespace dist {

namespace {

static TaskPtr old_task = nullptr;

bool RunSampleSubLoopOnce() {
  auto graph_pool = DistEngine::Get()->GetGraphPool();
  if (graph_pool->Full()) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    return true;
  }

  auto next_op = cuda::kDataCopy;
  auto next_q = dynamic_cast<MessageTaskQueue*>(DistEngine::Get()->GetTaskQueue(next_op));
  if (next_q->Full()) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    return true;
  }
  double shuffle_time, sample_time, get_miss_cache_index_time, send_time;

  Timer t_total;

  Timer t0;
  auto task = DoShuffle();
  if (task == nullptr) {
    LOG(FATAL) << "null task from DoShuffle!";
  }
  shuffle_time = t0.Passed();

#ifndef PIPELINE
  Timer t1;
  DoGPUSample(task);
  sample_time = t1.Passed();

  Timer t2;
#ifdef SAMGRAPH_LEGACY_CACHE_ENABLE
  DoGetCacheMissIndex(task);
#endif
  get_miss_cache_index_time = t2.Passed();

  LOG(DEBUG) << "RunSampleOnce next_q Send task";
  Timer t3;
  next_q->Send(task);
  send_time = t3.Passed();

  Profiler::Get().LogEpochAdd(task->key, kLogEpochSampleTime,
                              shuffle_time + sample_time);
  Profiler::Get().LogEpochAdd(task->key, KLogEpochSampleGetCacheMissIndexTime,
                              get_miss_cache_index_time);
  Profiler::Get().LogEpochAdd(task->key, kLogEpochSampleSendTime, send_time);
  Profiler::Get().LogEpochAdd(
      task->key, kLogEpochSampleTotalTime,
      shuffle_time + sample_time + get_miss_cache_index_time + send_time);
#else // PIPELINE

#pragma omp parallel num_threads(2)
  {
#pragma omp single
  {
#pragma omp task
    {
      LOG(DEBUG) << "RunSampleOnce next_q Send task";
      Timer t2;
      if(old_task != nullptr) {
        next_q->Send(old_task);
      }
      send_time = t2.Passed();
    }
#pragma omp task
    {

      Timer t1;
      DoGPUSample(task);
      DoGetCacheMissIndex(task);
      sample_time = t1.Passed();

      old_task = task;
      // if the last one
      auto shuffler = DistEngine::Get()->GetShuffler();
      if (shuffler->IsLastBatch()) {
        Timer t2;
        next_q->Send(task);
        send_time = t2.Passed();
        old_task = nullptr;
      }
    }
#pragma omp taskwait
  }
  }

  double total_time = t_total.Passed();
  Profiler::Get().LogEpochAdd(task->key, kLogEpochSampleTime,
                              total_time);

#endif // PIPELINE

  Profiler::Get().LogStep(task->key, kLogL1SampleTime,
                          shuffle_time + sample_time);
  Profiler::Get().LogStep(task->key, kLogL1SendTime,
                          send_time);
  Profiler::Get().LogStep(task->key, kLogL2ShuffleTime, shuffle_time);

  return true;
}

bool RunDataCopySubLoopOnce() {
  auto graph_pool = DistEngine::Get()->GetGraphPool();
  while (graph_pool->Full()) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    // return true;
  }

  auto this_op = cuda::kDataCopy;
  auto q = dynamic_cast<MessageTaskQueue*>(DistEngine::Get()->GetTaskQueue(this_op));
  Timer t4;
  auto task = q->Recv();
  double recv_time = t4.Passed();

  double extract_time=0,feat_copy_time=0;
  if (task) {
    Timer t0;
    DoGraphCopy(task);
    double graph_copy_time = t0.Passed();

    Timer t2;
#ifdef SAMGRAPH_LEGACY_CACHE_ENABLE
    DoCPUFeatureExtract(task);
    double extract_time = t2.Passed();

    Timer t3;
    DoFeatureCopy(task);
    double feat_copy_time = t3.Passed();
#endif
#ifdef SAMGRAPH_COLL_CACHE_VALIDATE
    auto backup_feat = Tensor::CopyTo(task->input_feat, DistEngine::Get()->GetTrainerCtx(), DistEngine::Get()->GetTrainerCopyStream());
    auto backup_label = Tensor::CopyTo(task->output_label, DistEngine::Get()->GetTrainerCtx(), DistEngine::Get()->GetTrainerCopyStream());
#endif
#ifdef SAMGRAPH_COLL_CACHE_ENABLE
    Timer t2_1;
    DoCollFeatLabelExtract(task);
    extract_time = t2_1.Passed();
#endif
#ifdef SAMGRAPH_COLL_CACHE_VALIDATE
    CollCacheManager::CheckCudaEqual(backup_feat->Data(), task->input_feat->Data(), backup_feat->NumBytes());
    CollCacheManager::CheckCudaEqual(backup_label->Data(), task->output_label->Data(), backup_label->NumBytes());
    LOG(INFO) << "Coll Cache Validate success " << task->key;
#endif

    LOG(DEBUG) << "Submit: process task with key " << task->key;
    graph_pool->Submit(task->key, task);

    Profiler::Get().LogStep(
        task->key, kLogL1CopyTime,
        recv_time + graph_copy_time + extract_time + feat_copy_time);
    Profiler::Get().LogStep(task->key, kLogL1RecvTime, recv_time);
    Profiler::Get().LogStep(task->key, kLogL2GraphCopyTime, graph_copy_time);
    Profiler::Get().LogStep(task->key, kLogL2ExtractTime, extract_time);
    Profiler::Get().LogStep(task->key, kLogL2FeatCopyTime, feat_copy_time);
    Profiler::Get().LogEpochAdd(
        task->key, kLogEpochCopyTime,
        recv_time + graph_copy_time + extract_time + feat_copy_time);
  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }

  return true;
}

bool RunCacheDataCopySubLoopOnce() {
  auto graph_pool = DistEngine::Get()->GetGraphPool();
  auto dist_type  = DistEngine::Get()->GetDistType();
  while (graph_pool->Full()) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    // return true;
  }

  auto this_op = cuda::kDataCopy;
  auto q = dynamic_cast<MessageTaskQueue*>(DistEngine::Get()->GetTaskQueue(this_op));
  // receive the task data from sample process
  Timer t4;
  auto task = q->Recv();
  double recv_time = t4.Passed();

  if (task) {
    Timer t0;
    DoGraphCopy(task);
    double graph_copy_time = t0.Passed();

    Timer t2;
#ifdef SAMGRAPH_LEGACY_CACHE_ENABLE
    switch(dist_type) {
      case (DistType::Extract): {
            DoCacheFeatureCopy(task);
            break;
      }
      case (DistType::Switch): {
            DoSwitchCacheFeatureCopy(task);
            break;
      }
      default:
            CHECK(0);
    }
    DoCPULabelExtractAndCopy(task);
#endif
#ifdef SAMGRAPH_COLL_CACHE_VALIDATE
    auto backup_feat = Tensor::CopyTo(task->input_feat, DistEngine::Get()->GetTrainerCtx(), DistEngine::Get()->GetTrainerCopyStream());
    auto backup_label = Tensor::CopyTo(task->output_label, DistEngine::Get()->GetTrainerCtx(), DistEngine::Get()->GetTrainerCopyStream());
#endif
#ifdef SAMGRAPH_COLL_CACHE_ENABLE
    DoCollFeatLabelExtract(task);
#endif
    double cache_feat_copy_time = t2.Passed();
#ifdef SAMGRAPH_COLL_CACHE_VALIDATE
    CollCacheManager::CheckCudaEqual(backup_feat->Data(), task->input_feat->Data(), backup_feat->NumBytes());
    CollCacheManager::CheckCudaEqual(backup_label->Data(), task->output_label->Data(), backup_label->NumBytes());
    LOG(INFO) << "Coll Cache Validate success " << task->key;
#endif

    LOG(DEBUG) << "Submit with cache: process task with key " << task->key;
    graph_pool->Submit(task->key, task);

    Profiler::Get().LogStep(task->key, kLogL1CopyTime,
                            recv_time + graph_copy_time + cache_feat_copy_time);
    Profiler::Get().LogStep(task->key, kLogL1RecvTime, recv_time);
    Profiler::Get().LogStep(task->key, kLogL2GraphCopyTime, graph_copy_time);
    Profiler::Get().LogStep(task->key, kLogL2CacheCopyTime,
                            cache_feat_copy_time);
    Profiler::Get().LogEpochAdd(
        task->key, kLogEpochCopyTime,
        recv_time + graph_copy_time + cache_feat_copy_time);
  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }
  return true;
}

void DataCopySubLoop(int count) {
  LoopOnceFunction func;
  if (!RunConfig::UseGPUCache()) {
    func = RunDataCopySubLoopOnce;
  } else {
    func = RunCacheDataCopySubLoopOnce;
  }

  while ((count--) && !DistEngine::Get()->ShouldShutdown() && func());

  DistEngine::Get()->ReportThreadFinish();
}

} // namespace

void RunArch5LoopsOnce(DistType dist_type) {
  if (dist_type == DistType::Sample) {
    LOG(DEBUG) << "RunArch5LoopsOnce with Sample!";
    RunSampleSubLoopOnce();
  }
  else if (dist_type == DistType::Extract || dist_type == DistType::Switch) {
    if (!RunConfig::UseGPUCache()) {
      LOG(DEBUG) << "RunArch5LoopsOnce with Extract no Cache!";
      RunDataCopySubLoopOnce();
    } else {
      LOG(DEBUG) << "RunArch5LoopsOnce with Extract Cache!";
      RunCacheDataCopySubLoopOnce();
    }
  } else {
    LOG(FATAL) << "dist type is illegal!";
  }
}

ExtractFunction GetArch5Loops() {
  return DataCopySubLoop;
}

}  // namespace dist
}  // namespace common
}  // namespace samgraph
