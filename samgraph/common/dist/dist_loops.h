#pragma once

#ifndef SAMGRAPH_DIST_LOOPS_H
#define SAMGRAPH_DIST_LOOPS_H

#include "../common.h"
#include "dist_engine.h"

namespace samgraph {
namespace common {
namespace dist {

void RunArch5LoopsOnce(DistType dist_type);

// common steps
TaskPtr DoShuffle();
void DoGPUSample(TaskPtr task);
void DoGraphCopy(TaskPtr task);
void DoIdCopy(TaskPtr task);
void DoCPUFeatureExtract(TaskPtr task);
void DoFeatureCopy(TaskPtr task);

}  // namespace dist
}  // namespace common
}  // namespace samgraph

#endif // SAMGRAPH_DIST_LOOPS_H
