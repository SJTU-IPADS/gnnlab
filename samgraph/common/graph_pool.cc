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

#include "graph_pool.h"

#include <thread>

#include "common.h"
#include "constant.h"
#include "logging.h"

namespace samgraph {
namespace common {

GraphPool::~GraphPool() { _stop = true; }

std::shared_ptr<GraphBatch> GraphPool::GetGraphBatch() {
  while (true) {
    {
      std::lock_guard<std::mutex> lock(_mutex);
      if (!_pool.empty()) {
        auto batch = _pool.front();
        _pool.pop();
        auto key = batch->key;
        LOG(DEBUG) << "GraphPool: Get batch with key " << key;
        return batch;
      } else if (_stop) {
        return nullptr;
      }
    }
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }

  return nullptr;
}

void GraphPool::Submit(uint64_t key, std::shared_ptr<GraphBatch> batch) {
  std::lock_guard<std::mutex> lock(_mutex);
  CHECK(!_stop);
  _pool.push(batch);

  LOG(DEBUG) << "GraphPool: Add batch with key " << key;
}

bool GraphPool::Full() {
  std::lock_guard<std::mutex> lock(_mutex);
  return _pool.size() >= _max_size;
}

}  // namespace common
}  // namespace samgraph
