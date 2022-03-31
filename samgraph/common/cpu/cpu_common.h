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

#ifndef SAMGRAPH_CPU_COMMON_H
#define SAMGRAPH_CPU_COMMON_H

namespace samgraph {
namespace common {
namespace cpu {

enum CPUHashType { kCPUHash0 = 0, kCPUHash1, kCPUHash2 };

}
}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_CPU_COMMON_H