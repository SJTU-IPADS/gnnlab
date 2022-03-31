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

#include <unordered_set>

#include "../common.h"
#include "../constant.h"
#include "../logging.h"
#include "cpu_function.h"

namespace samgraph {
namespace common {
namespace cpu {

void CPUSanityCheckList(const IdType *input, size_t num_input,
                        IdType invalid_val) {
  for (size_t i = 0; i < num_input; i++) {
    CHECK_NE(input[i], invalid_val);
  }
}

void CPUSanityCheckNoDuplicate(const IdType *input, size_t num_input) {
  std::unordered_set<IdType> visited_elem;
  for (size_t i = 0; i < num_input; i++) {
    if (visited_elem.count(input[i]) > 0) {
      LOG(DEBUG) << "duplicate" << input[i];
      CHECK(false);
    }
    visited_elem.insert(input[i]);
  }
}

}  // namespace cpu
}  // namespace common
}  // namespace samgraph