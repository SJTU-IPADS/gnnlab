#ifndef SAMGRAPH_CUDA_LOOPS_H
#define SAMGRAPH_CUDA_LOOPS_H

#include <vector>

#include "../common.h"

namespace samgraph {
namespace common {
namespace cuda {

void RunArch1LoopsOnce();
void RunArch2LoopsOnce();
void RunArch3LoopsOnce();

std::vector<LoopFunction> GetArch1Loops();
std::vector<LoopFunction> GetArch2Loops();
std::vector<LoopFunction> GetArch3Loops();

// Common steps
TaskPtr DoShuffle();
void DoGPUSample(TaskPtr task);
void DoGraphCopy(TaskPtr task);
void DoIdCopy(TaskPtr task);
void DoCacheIdCopy(TaskPtr task);
void DoCPUFeatureExtract(TaskPtr task);
void DoGPUFeatureExtract(TaskPtr task);
void DoFeatureCopy(TaskPtr task);
void DoCacheFeatureCopy(TaskPtr task);

}  // namespace cuda
}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_CUDA_LOOPS_H