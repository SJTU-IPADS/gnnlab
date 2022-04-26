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
void RunArch4LoopsOnce();
void RunArch7LoopsOnce();

std::vector<LoopFunction> GetArch1Loops();
std::vector<LoopFunction> GetArch2Loops();
std::vector<LoopFunction> GetArch3Loops();
std::vector<LoopFunction> GetArch4Loops();

// common steps
TaskPtr DoShuffle();
void DoGPUSample(TaskPtr task);
void DoGPUSampleDyCache(TaskPtr task, std::function<void(TaskPtr)> &neighbour_cb); 
void DoGPUSampleAllNeighbour(TaskPtr task);
void DoGraphCopy(TaskPtr task);
void DoIdCopy(TaskPtr task);
void DoCacheIdCopy(TaskPtr task);
void DoCacheIdCopyToCPU(TaskPtr task);
void DoCPUFeatureExtract(TaskPtr task);
void DoGPUFeatureExtract(TaskPtr task);
void DoGPULabelExtract(TaskPtr task);
void DoCPULabelExtractAndCopy(TaskPtr task);
void DoFeatureCopy(TaskPtr task);
void DoCacheFeatureCopy(TaskPtr task);
void DoDynamicCacheFeatureCopy(TaskPtr task);

void StatisticUMSampleCacheHit(int task_key, int layer,const IdType* input, size_t num_input, Context input_ctx);

}  // namespace cuda
}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_CUDA_LOOPS_H