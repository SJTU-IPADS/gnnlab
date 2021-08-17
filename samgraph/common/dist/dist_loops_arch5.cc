#include "../cuda/cuda_loops.h"

#include <chrono>
#include <numeric>

#include "../device.h"
#include "../function.h"
#include "../logging.h"
#include "../profiler.h"
#include "../run_config.h"
#include "../timer.h"

#include "dist_engine.h"
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

bool RunSampleSubLoopOnce() {
  auto graph_pool = DistEngine::Get()->GetGraphPool();
  if (graph_pool->Full()) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    return true;
  }

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

    // TODO: implement function "sendTask" to send task
    next_q->AddTask(task);
    // XXX: remove this submission
    LOG(DEBUG) << "Submit: process task with key " << task->key;
    graph_pool->Submit(task->key, task);

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
  auto graph_pool = DistEngine::Get()->GetGraphPool();
  if (graph_pool->Full()) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    return true;
  }

  auto this_op = cuda::kDataCopy;
  auto q = DistEngine::Get()->GetTaskQueue(this_op);
  // TODO: implement function "recvTask" to receive the task
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

} // namespace

// TODO: split the sampling and extracting
void RunArch5LoopsOnce() {
  RunSampleSubLoopOnce();
  if (!RunConfig::UseGPUCache()) {
    // RunDataCopySubLoopOnce();
  } else {
    // TODO: implement function RunCacheDataCopySubLoopOnce
    LOG(FATAL) << "RunCacheDataCopySubLoopOnce needs to implement!";
  }

}

}  // namespace dist
}  // namespace common
}  // namespace samgraph
