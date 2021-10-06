#ifndef SAMGRAPH_CPU_LOOPS_H
#define SAMGRAPH_CPU_LOOPS_H

#include "../common.h"

namespace samgraph {
namespace common {
namespace cpu {

void RunArch0LoopsOnce();

std::vector<LoopFunction> GetArch0Loops();

// common steps
TaskPtr DoShuffle();
void DoCPUSample(TaskPtr task);
void DoGraphCopy(TaskPtr task);
void DoFeatureExtract(TaskPtr task);
void DoFeatureCopy(TaskPtr task);

void DoCacheIdCopy(TaskPtr task);
void DoGPULabelExtract(TaskPtr task);
void DoCPULabelExtractAndCopy(TaskPtr task);
void DoCacheFeatureExtractCopy(TaskPtr task);

}  // namespace cpu
}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_CPU_LOOPS_H