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

#include <thread>

#include "../logging.h"
#include "../profiler.h"
#include "../timer.h"
#include "dist_engine.h"
#include "dist_loops.h"

/* clang-format off
 * +--------------------------------------------------------------+
 * |                                                              |
 * |                                                              |
 * | Sampling----------------+          +--------------> Training |
 * |                         |          |                         |
 * |                         |          |                         |
 * |                         |   GPU    |                         |
 * +-------------------------|----------|-------------------------+
 *                           |          |                          
 *                           |          |                          
 *                           |          |                          
 *                      +----v----------|----+                     
 *                      |                    |                     
 *                      | Feature Extraction |                     
 *                      |                    |                     
 *                      |                    |                     
 *                      |                    |                     
 *                      |                    |                     
 *                      |        CPU         |                     
 *                      +--------------------+                     
 * clang-format on
 */

namespace samgraph {
namespace common {
namespace dist {

namespace {
bool RunSampleSubLoopOnce() {
  auto next_op = cuda::kDataCopy;
  auto next_q = DistEngine::Get()->GetTaskQueue(next_op);
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

    Profiler::Get().LogStep(task->key, kLogL1SampleTime,
                            shuffle_time + sample_time);
    Profiler::Get().LogStep(task->key, kLogL2ShuffleTime, shuffle_time);
    Profiler::Get().LogEpochAdd(task->key, kLogEpochSampleTime,
                                shuffle_time + sample_time);
    Profiler::Get().LogEpochAdd(task->key, kLogEpochSampleTotalTime,
                                shuffle_time + sample_time);
  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }

  return true;
}

bool RunDataCopySubLoopOnce() {
  auto graph_pool = DistEngine::Get()->GetGraphPool();
  if (graph_pool->Full()) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    return true;
  }

  auto this_op = cuda::kDataCopy;
  auto this_q = DistEngine::Get()->GetTaskQueue(this_op);
  auto task = this_q->GetTask();

  double id_copy_time=0,extract_time=0,feat_copy_time=0;
  if (task) {
#ifdef SAMGRAPH_LEGACY_CACHE_ENABLE
    Timer t2;
    DoIdCopy(task);
    double id_copy_time = t2.Passed();

    Timer t3;
    DoCPUFeatureExtract(task);
    double extract_time = t3.Passed();

    Timer t4;
    DoFeatureCopy(task);
    double feat_copy_time = t4.Passed();
#endif
#ifdef SAMGRAPH_COLL_CACHE_VALIDATE
    auto backup_feat = Tensor::CopyTo(task->input_feat, DistEngine::Get()->GetTrainerCtx(), DistEngine::Get()->GetTrainerCopyStream());
    auto backup_label = Tensor::CopyTo(task->output_label, DistEngine::Get()->GetTrainerCtx(), DistEngine::Get()->GetTrainerCopyStream());
#endif
#ifdef SAMGRAPH_COLL_CACHE_ENABLE
    Timer t3_1;
    DoCollFeatLabelExtract(task);
    extract_time = t3_1.Passed();
#endif
#ifdef SAMGRAPH_COLL_CACHE_VALIDATE
    CollCacheManager::CheckCudaEqual(backup_feat->Data(), task->input_feat->Data(), backup_feat->NumBytes());
    CollCacheManager::CheckCudaEqual(backup_label->Data(), task->output_label->Data(), backup_label->NumBytes());
#endif

    LOG(DEBUG) << "Submit: process task with key " << task->key;
    graph_pool->Submit(task->key, task);

    Profiler::Get().LogStep(task->key, kLogL1CopyTime,
                            id_copy_time + extract_time + feat_copy_time);
    Profiler::Get().LogStep(task->key, kLogL2IdCopyTime, id_copy_time);
    Profiler::Get().LogStep(task->key, kLogL2ExtractTime, extract_time);
    Profiler::Get().LogStep(task->key, kLogL2FeatCopyTime, feat_copy_time);
    Profiler::Get().LogEpochAdd(task->key, kLogEpochCopyTime,
                                id_copy_time + extract_time + feat_copy_time);
  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }

  return true;
}

bool RunCacheDataCopySubLoopOnce() {
  auto graph_pool = DistEngine::Get()->GetGraphPool();
  if (graph_pool->Full()) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    return true;
  }

  auto this_op = cuda::kDataCopy;
  auto this_q = DistEngine::Get()->GetTaskQueue(this_op);
  auto task = this_q->GetTask();

  double get_miss_cache_index_time=0,id_copy_time=0,feat_copy_time=0;
  if (task) {
#ifdef SAMGRAPH_LEGACY_CACHE_ENABLE
    Timer t2;
    DoArch6GetCacheMissIndex(task);
    double get_miss_cache_index_time = t2.Passed();

    Timer t3;
    DoCacheIdCopyToCPU(task);
    double id_copy_time = t3.Passed();

    Timer t4;
    DoCPULabelExtractAndCopy(task);
    DoArch6CacheFeatureCopy(task);
    double feat_copy_time = t4.Passed();
#endif
#ifdef SAMGRAPH_COLL_CACHE_VALIDATE
    auto backup_feat = Tensor::CopyTo(task->input_feat, DistEngine::Get()->GetTrainerCtx(), DistEngine::Get()->GetTrainerCopyStream());
    auto backup_label = Tensor::CopyTo(task->output_label, DistEngine::Get()->GetTrainerCtx(), DistEngine::Get()->GetTrainerCopyStream());
#endif
#ifdef SAMGRAPH_COLL_CACHE_ENABLE
    Timer t3_1;
    DoCollFeatLabelExtract(task);
    feat_copy_time = t3_1.Passed();
#endif
#ifdef SAMGRAPH_COLL_CACHE_VALIDATE
    CollCacheManager::CheckCudaEqual(backup_feat->Data(), task->input_feat->Data(), backup_feat->NumBytes());
    CollCacheManager::CheckCudaEqual(backup_label->Data(), task->output_label->Data(), backup_label->NumBytes());
#endif

    LOG(DEBUG) << "Submit with cache: process task with key " << task->key;
    graph_pool->Submit(task->key, task);

    Profiler::Get().LogEpochAdd(task->key, KLogEpochSampleGetCacheMissIndexTime,
                                get_miss_cache_index_time);
    Profiler::Get().LogEpochAdd(task->key, kLogEpochSampleTotalTime,
                                get_miss_cache_index_time);
    Profiler::Get().LogStep(task->key, kLogL1CopyTime,
                            id_copy_time + feat_copy_time);
    Profiler::Get().LogStep(task->key, kLogL2IdCopyTime, id_copy_time);
    Profiler::Get().LogStep(task->key, kLogL2FeatCopyTime, feat_copy_time);
    Profiler::Get().LogEpochAdd(task->key, kLogEpochCopyTime,
                                id_copy_time + feat_copy_time);
  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }

  return true;
}

void SampleSubLoop() {
  while (RunSampleSubLoopOnce() && !DistEngine::Get()->ShouldShutdown()) {
  }
  DistEngine::Get()->ReportThreadFinish();
}

void DataCopySubLoop() {
  LoopOnceFunction func;
  if (!RunConfig::UseGPUCache()) {
    func = RunDataCopySubLoopOnce;
  } else {
    func = RunCacheDataCopySubLoopOnce;
  }

  while (func() && !DistEngine::Get()->ShouldShutdown()) {
  }

  DistEngine::Get()->ReportThreadFinish();
}

void SampleCopySubSloop() {
  while (true) {
    if (DistEngine::Get()->ShouldShutdown()) break;
    RunSampleSubLoopOnce();
    if (!RunConfig::UseGPUCache()) {
      RunDataCopySubLoopOnce();
    } else {
      RunCacheDataCopySubLoopOnce();
    }
  }
  DistEngine::Get()->ReportThreadFinish();
}

}  // namespace

void RunArch6LoopsOnce() {
  RunSampleSubLoopOnce();
  if (!RunConfig::UseGPUCache()) {
    RunDataCopySubLoopOnce();
  } else {
    RunCacheDataCopySubLoopOnce();
  }
}

std::vector<LoopFunction> GetArch6Loops() {
  std::vector<LoopFunction> func;

  /** make sample & extract overlap or not */
  // func.push_back(SampleSubLoop);
  // func.push_back(DataCopySubLoop);
  func.push_back(SampleCopySubSloop);

  return func;
}

}  // namespace dist
}  // namespace common
}  // namespace samgraph