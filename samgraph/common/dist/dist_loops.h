#pragma once

#ifndef SAMGRAPH_DIST_LOOPS_H
#define SAMGRAPH_DIST_LOOPS_H

#include "../common.h"
#include "dist_engine.h"

namespace samgraph {
namespace common {
namespace dist {

void RunArch5LoopsOnce(DistType dist_type);
void RunArch6LoopsOnce();

// common steps
TaskPtr DoShuffle();
void DoGPUSample(TaskPtr task);
void DoGetCacheMissIndex(TaskPtr task);
void DoGraphCopy(TaskPtr task);
void DoIdCopy(TaskPtr task);
void DoCPUFeatureExtract(TaskPtr task);
void DoFeatureCopy(TaskPtr task);

void DoCacheIdCopy(TaskPtr task);
void DoCacheIdCopyToCPU(TaskPtr task);
void DoSwitchCacheFeatureCopy(TaskPtr task);
void DoCacheFeatureCopy(TaskPtr task);
void DoGPULabelExtract(TaskPtr task);
void DoCPULabelExtractAndCopy(TaskPtr task);

void DoArch6GetCacheMissIndex(TaskPtr task);
void DoArch6CacheFeatureCopy(TaskPtr task);

typedef void (*ExtractFunction)(int);
ExtractFunction GetArch5Loops();
std::vector<LoopFunction> GetArch6Loops();

}  // namespace dist
}  // namespace common
}  // namespace samgraph

#endif // SAMGRAPH_DIST_LOOPS_H
