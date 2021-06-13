#include <thread>

#include "../logging.h"
#include "../profiler.h"
#include "../timer.h"
#include "cuda_engine.h"
#include "cuda_loops.h"
#include "cuda_loops_common.h"

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

namespace {}  // namespace

void RunOffloadLoopOnce() {}

std::vector<LoopFunction> GetOffloadLoops() {}

}  // namespace cuda
}  // namespace common
}  // namespace samgraph