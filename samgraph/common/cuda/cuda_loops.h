#ifndef SAMGRAPH_CUDA_LOOPS_H
#define SAMGRAPH_CUDA_LOOPS_H

#include <vector>

#include "../common.h"

namespace samgraph {
namespace common {
namespace cuda {

void RunDedicatedLoopOnce();
void RunStandaloneLoopOnce();
void RunOffloadLoopOnce();

std::vector<LoopFunction> GetDedicatedLoops();
std::vector<LoopFunction> GetStandaloneLoops();
std::vector<LoopFunction> GetOffloadLoops();

}  // namespace cuda
}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_CUDA_LOOPS_H