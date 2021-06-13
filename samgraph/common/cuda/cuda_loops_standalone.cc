#include <thread>

#include "../profiler.h"
#include "../timer.h"
#include "cuda_engine.h"
#include "cuda_loops.h"
#include "cuda_loops_common.h"

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
bool RunGPUSampleCopyLoopOnce() {
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

    Profiler::Get().Log(task->key, kLogL1SampleTime,
                        shuffle_time + sample_time + extract_time);
    Profiler::Get().Log(task->key, kLogL2ShuffleTime, shuffle_time);
    Profiler::Get().Log(task->key, kLogL2ExtractTime, extract_time);
  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }

  return true;
}

void StandaloneLoop() {
  while (RunGPUSampleCopyLoopOnce() && !GPUEngine::Get()->ShouldShutdown()) {
  }
  GPUEngine::Get()->ReportThreadFinish();
}

}  // namespace

void RunStandaloneLoopOnce() { RunGPUSampleCopyLoopOnce(); }

std::vector<LoopFunction> GetStandaloneLoops() {
  std::vector<LoopFunction> func;
  func.push_back(StandaloneLoop);
  return func;
}

}  // namespace cuda
}  // namespace common
}  // namespace samgraph