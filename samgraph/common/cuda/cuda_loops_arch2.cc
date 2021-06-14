#include <thread>

#include "../logging.h"
#include "../profiler.h"
#include "../timer.h"
#include "cuda_engine.h"
#include "cuda_loops.h"

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
namespace cuda {

namespace {

bool RunSampleSubLoopOnce() {
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

bool RunDataCopySubLoopOnce() {
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

}  // namespace

void RunArch2LoopsOnce() {
  RunSampleSubLoopOnce();
  RunDataCopySubLoopOnce();
}

std::vector<LoopFunction> GetArch2Loops() {
  CHECK(0) << "arch2 doesn't support background execution";
}

}  // namespace cuda
}  // namespace common
}  // namespace samgraph