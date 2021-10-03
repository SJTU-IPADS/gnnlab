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

bool RunSampleCopySubLoopOnce() {
  auto graph_pool = DistEngine::Get()->GetGraphPool();
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
    DoIdCopy(task);
    double id_copy_time = t2.Passed();

    Timer t3;
    DoCPUFeatureExtract(task);
    double extract_time = t3.Passed();

    Timer t4;
    DoFeatureCopy(task);
    double feat_copy_time = t4.Passed();

    LOG(DEBUG) << "Submit with cache: process task with key " << task->key;
    graph_pool->Submit(task->key, task);

    Profiler::Get().LogStep(task->key, kLogL1SampleTime,
                            shuffle_time + sample_time);
    Profiler::Get().LogStep(task->key, kLogL2ShuffleTime, shuffle_time);
    Profiler::Get().LogEpochAdd(task->key, kLogEpochSampleTime,
                                shuffle_time + sample_time);
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

bool RunCacheSampleCopySubLoopOnce() {
  auto graph_pool = DistEngine::Get()->GetGraphPool();
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

    Timer t4;
    DoGPULabelExtract(task);
    DoGetCacheMissIndexAndFeatureCopy(task);
    double feat_copy_time = t4.Passed();

    LOG(DEBUG) << "Submit with cache: process task with key " << task->key;
    graph_pool->Submit(task->key, task);

    Profiler::Get().LogStep(task->key, kLogL1SampleTime,
                            shuffle_time + sample_time);
    Profiler::Get().LogStep(task->key, kLogL2ShuffleTime, shuffle_time);
    Profiler::Get().LogEpochAdd(task->key, kLogEpochSampleTime,
                                shuffle_time + sample_time);
    Profiler::Get().LogStep(task->key, kLogL1CopyTime, feat_copy_time);
    Profiler::Get().LogStep(task->key, kLogL2FeatCopyTime, feat_copy_time);
    Profiler::Get().LogEpochAdd(task->key, kLogEpochCopyTime, feat_copy_time);
  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }

  return true;
}

void RunArch6LoopsOnce() {
  if (!RunConfig::UseGPUCache()) {
    RunSampleCopySubLoopOnce();
  } else {
    RunCacheSampleCopySubLoopOnce();
  }
}

}  // namespace dist
}  // namespace common
}  // namespace samgraph