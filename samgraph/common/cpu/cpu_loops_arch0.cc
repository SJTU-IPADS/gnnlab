#include <thread>

#include "../profiler.h"
#include "../timer.h"
#include "cpu_engine.h"
#include "cpu_loops.h"

namespace samgraph {
namespace common {
namespace cpu {

namespace {
bool RunSampleSubLoopOnce() {
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

    Profiler::Get().Log(task->key, kLogL1SampleTime,
                        shuffle_time + sample_time);
    Profiler::Get().Log(task->key, kLogL2ShuffleTime, shuffle_time);
    Profiler::Get().Log(task->key, kLogL1CopyTime,
                        extract_time + graph_copy_time + feat_copy_time);
    Profiler::Get().Log(task->key, kLogL2ExtractTime, extract_time);
    Profiler::Get().Log(task->key, kLogL2GraphCopyTime, graph_copy_time);
    Profiler::Get().Log(task->key, kLogL2FeatCopyTime, feat_copy_time);

    LOG(DEBUG) << "CPUSampleLoop: process task with key " << task->key;
  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }

  return true;
}

void SampleSubLoop() {
  while (RunSampleSubLoopOnce() && !CPUEngine::Get()->ShouldShutdown()) {
  }
  CPUEngine::Get()->ReportThreadFinish();
}

}  // namespace

void RunArch0LoopsOnce() { RunSampleSubLoopOnce(); }

std::vector<LoopFunction> GetArch0Loops() {
  std::vector<LoopFunction> func;

  func.push_back(SampleSubLoop);

  return func;
}

}  // namespace cpu
}  // namespace common
}  // namespace samgraph