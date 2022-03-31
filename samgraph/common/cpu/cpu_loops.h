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