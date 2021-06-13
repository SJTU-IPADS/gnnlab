#ifndef SAMGRAPH_LOOPS_COMMON_H
#define SAMGRAPH_LOOPS_COMMON_H

namespace samgraph {
namespace common {
namespace cuda {

TaskPtr DoShuffle();
void DoGPUSample(Taskptr task);
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

#endif  // SAMGRAPH_LOOPS_COMMON_H