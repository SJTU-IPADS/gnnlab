/*
 * Copyright 2022 Institute of Parallel and Distributed Systems, Shanghai Jiao Tong University
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

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
void RunArch9LoopsOnce(DistType dist_type);

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
ExtractFunction GetArch9Loops();

}  // namespace dist
}  // namespace common
}  // namespace samgraph

#endif // SAMGRAPH_DIST_LOOPS_H
