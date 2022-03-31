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
#include "cuda_engine.h"
#include "cuda_loops.h"

/* clang-format off
 *  +----------------------------------------------------------------+
 *  |                                                                |
 *  |                                                                |
 *  | Sampling -----------> Feature Extraction ----------> Training  |
 *  |                                                                |
 *  |                                                                |
 *  |                             GPU                                |
 *  +----------------------------------------------------------------+
 * clang-format on
 */

namespace samgraph {
namespace common {
namespace cuda {

namespace {

bool RunSampleCopySubLoopOnce() {
  auto graph_pool = GPUEngine::Get()->GetGraphPool();
  if (graph_pool->Full()) {
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

    Timer t2;
    DoGPUFeatureExtract(task);
    double extract_time = t2.Passed();

    LOG(DEBUG) << "Submit: process task with key " << task->key;
    graph_pool->Submit(task->key, task);

    Profiler::Get().LogStep(task->key, kLogL1SampleTime,
                            shuffle_time + sample_time);
    Profiler::Get().LogStep(task->key, kLogL1CopyTime, extract_time);
    Profiler::Get().LogStep(task->key, kLogL2ShuffleTime, shuffle_time);
    Profiler::Get().LogStep(task->key, kLogL2ExtractTime, extract_time);
    Profiler::Get().LogEpochAdd(task->key, kLogEpochSampleTime,
                                shuffle_time + sample_time);
    Profiler::Get().LogEpochAdd(task->key, kLogEpochCopyTime, extract_time);
  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }

  return true;
}

}  // namespace

void RunArch1LoopsOnce() { RunSampleCopySubLoopOnce(); }

std::vector<LoopFunction> GetArch1Loops() {
  CHECK(0) << "arch1 doesn't support background execution";
  return {};
}

}  // namespace cuda
}  // namespace common
}  // namespace samgraph