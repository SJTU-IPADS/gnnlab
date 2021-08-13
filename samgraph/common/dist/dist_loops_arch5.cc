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
  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }

  return true;
}

} // namespace

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
