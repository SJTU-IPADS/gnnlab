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
#include "cuda_loops_common.h"

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

bool RunGPUSampleLoopOnce() {
  auto next_op = kDataCopy;
  auto next_q = GPUEngine::Get()->GetTaskQueue(next_op);
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

    Profiler::Get().Log(task->key, kLogL1SampleTime,
                        shuffle_time + sample_time);
    Profiler::Get().Log(task->key, kLogL2ShuffleTime, shuffle_time);
  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }

  return true;
}

bool RunDataCopyLoopOnce() {
  auto graph_pool = GPUEngine::Get()->GetGraphPool();
  if (graph_pool->Full()) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    return true;
  }

  auto this_op = kDataCopy;
  auto q = GPUEngine::Get()->GetTaskQueue(this_op);
  auto task = q->GetTask();

  if (task) {
    Timer t0;
    DoGraphCopy(task);
    double graph_copy_time = t0.Passed();

    Timer t1;
    DoIdCopy(task);
    double id_copy_time = t1.Passed();

    Timer t2;
    DoCPUFeatureExtract(task);
    double extract_time = t2.Passed();

    Timer t3;
    DoFeatureCopy(task);
    double feat_copy_time = t3.Passed();

    LOG(DEBUG) << "Submit: process task with key " << task->key;
    graph_pool->Submit(task->key, task);

    Profiler::Get().Log(
        task->key, kLogL1CopyTime,
        graph_copy_time + id_copy_time + extract_time + feat_copy_time);
    Profiler::Get().Log(task->key, kLogL2GraphCopyTime, graph_copy_time);
    Profiler::Get().Log(task->key, kLogL2IdCopyTime, id_copy_time);
    Profiler::Get().Log(task->key, kLogL2ExtractTime, extract_time);
    Profiler::Get().Log(task->key, kLogL2FeatCopyTime, feat_copy_time);

  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }

  return true;
}

bool RunCacheDataCopyLoopOnce() {
  auto graph_pool = GPUEngine::Get()->GetGraphPool();
  if (graph_pool->Full()) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    return true;
  }

  auto this_op = kDataCopy;
  auto q = GPUEngine::Get()->GetTaskQueue(this_op);
  auto task = q->GetTask();

  if (task) {
    Timer t0;
    DoGraphCopy(task);
    double graph_copy_time = t0.Passed();

    Timer t1;
    DoCacheIdCopy(task);
    double id_copy_time = t1.Passed();

    Timer t2;
    DoCacheFeatureCopy(task);
    double cache_feat_copy_time = t2.Passed();

    LOG(DEBUG) << "Submit with cache: process task with key " << task->key;
    graph_pool->Submit(task->key, task);

    Profiler::Get().Log(task->key, kLogL1CopyTime,
                        graph_copy_time + id_copy_time + cache_feat_copy_time);
    Profiler::Get().Log(task->key, kLogL2GraphCopyTime, graph_copy_time);
    Profiler::Get().Log(task->key, kLogL2IdCopyTime, id_copy_time);
    Profiler::Get().Log(task->key, kLogL2CacheCopyTime, cache_feat_copy_time);
  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }
  return true;
}

void GPUSampleLoop() {
  while (RunGPUSampleLoopOnce() && !GPUEngine::Get()->ShouldShutdown()) {
  }
  GPUEngine::Get()->ReportThreadFinish();
}

void DataCopyLoop() {
  LoopOnceFunction func;
  if (!RunConfig::UseGPUCache()) {
    func = RunDataCopyLoopOnce;
  } else {
    func = RunCacheDataCopyLoopOnce;
  }

  while (func() && !GPUEngine::Get()->ShouldShutdown()) {
  }

  GPUEngine::Get()->ReportThreadFinish();
}

}  // namespace

}  // namespace cuda
}  // namespace common
}  // namespace samgraph
