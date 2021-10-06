#include <thread>

#include "../profiler.h"
#include "../run_config.h"
#include "../timer.h"
#include "cpu_engine.h"
#include "cpu_loops.h"

namespace samgraph {
namespace common {
namespace cpu {

/* clang-format off
 *  +-------------------------+          +----------------------+
 *  |                         |          |                      |
 *  |                         |          |                      |
 *  |                         |          |                      |
 *  |  Sampling + Extracting  ----------->       Training       |
 *  |                         |          |                      |
 *  |                         |          |                      |
 *  |          CPU            |          |         GPU          |
 *  +-------------------------+          +----------------------+
 * clang-format on
 */

namespace {
bool RunSampleCopySubLoopOnce() {
  auto graph_pool = CPUEngine::Get()->GetGraphPool();
  if (graph_pool->Full()) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    return true;
  }

  Timer t0;
  auto task = DoShuffle();
  if (task) {
    double shuffle_time = t0.Passed();

    Timer t1;
    DoCPUSample(task);
    double sample_time = t1.Passed();

    Timer t2;
    DoFeatureExtract(task);
    double extract_time = t2.Passed();

    Timer t3;
    DoGraphCopy(task);
    double graph_copy_time = t3.Passed();

    Timer t4;
    DoFeatureCopy(task);
    double feat_copy_time = t4.Passed();

    graph_pool->Submit(task->key, task);

    Profiler::Get().LogStep(task->key, kLogL1SampleTime,
                            shuffle_time + sample_time);
    Profiler::Get().LogStep(task->key, kLogL2ShuffleTime, shuffle_time);
    Profiler::Get().LogStep(task->key, kLogL1CopyTime,
                            extract_time + graph_copy_time + feat_copy_time);
    Profiler::Get().LogStep(task->key, kLogL2ExtractTime, extract_time);
    Profiler::Get().LogStep(task->key, kLogL2GraphCopyTime, graph_copy_time);
    Profiler::Get().LogStep(task->key, kLogL2FeatCopyTime, feat_copy_time);
    Profiler::Get().LogEpochAdd(task->key, kLogEpochSampleTime,
                                shuffle_time + sample_time);
    Profiler::Get().LogEpochAdd(
        task->key, kLogEpochCopyTime,
        extract_time + graph_copy_time + feat_copy_time);
    LOG(DEBUG) << "CPUSampleLoop: process task with key " << task->key;
  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }

  return true;
}

bool RunCacheSampleCopySubLoopOnce() {
  auto graph_pool = CPUEngine::Get()->GetGraphPool();
  if (graph_pool->Full()) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    return true;
  }

  Timer t0;
  auto task = DoShuffle();
  if (task) {
    double shuffle_time = t0.Passed();

    Timer t1;
    DoCPUSample(task);
    double sample_time = t1.Passed();

    Timer t2;
    DoGraphCopy(task);
    double graph_copy_time = t2.Passed();

    Timer t3;
    DoCacheIdCopy(task);
    double id_copy_time = t3.Passed();

    Timer t4;
    DoCPULabelExtractAndCopy(task);
    DoCacheFeatureExtractCopy(task);
    double feat_copy_time = t4.Passed();

    graph_pool->Submit(task->key, task);

    Profiler::Get().LogStep(task->key, kLogL1SampleTime,
                            shuffle_time + sample_time);
    Profiler::Get().LogStep(task->key, kLogL2ShuffleTime, shuffle_time);
    Profiler::Get().LogStep(task->key, kLogL1CopyTime,
                            graph_copy_time + id_copy_time + feat_copy_time);
    Profiler::Get().LogStep(task->key, kLogL2GraphCopyTime, graph_copy_time);
    Profiler::Get().LogStep(task->key, kLogL2IdCopyTime, id_copy_time);
    Profiler::Get().LogStep(task->key, kLogL2CacheCopyTime, feat_copy_time);
    Profiler::Get().LogEpochAdd(task->key, kLogEpochSampleTime,
                                shuffle_time + sample_time);
    Profiler::Get().LogEpochAdd(
        task->key, kLogEpochCopyTime,
        graph_copy_time + id_copy_time + feat_copy_time);
    LOG(DEBUG) << "CPUSampleLoop: process task with key " << task->key;
  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }

  return true;
}

void SampleCopySubLoop() {
  if (RunConfig::UseGPUCache()) {
    while (RunSampleCopySubLoopOnce() && !CPUEngine::Get()->ShouldShutdown()) {
    }
  } else {
    while (RunCacheSampleCopySubLoopOnce() &&
           !CPUEngine::Get()->ShouldShutdown()) {
    }
  }
  CPUEngine::Get()->ReportThreadFinish();
}

}  // namespace

void RunArch0LoopsOnce() {
  if (!RunConfig::UseGPUCache()) {
    RunSampleCopySubLoopOnce();
  } else {
    RunCacheSampleCopySubLoopOnce();
  }
}

std::vector<LoopFunction> GetArch0Loops() {
  std::vector<LoopFunction> func;

  func.push_back(SampleCopySubLoop);

  return func;
}

}  // namespace cpu
}  // namespace common
}  // namespace samgraph