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

#ifndef SAMGRAPH_WORKSPACE_POOL_H
#define SAMGRAPH_WORKSPACE_POOL_H

#include <array>
#include <memory>
#include <mutex>

#include "common.h"
#include "device.h"

namespace samgraph {
namespace common {

class WorkspacePool {
 public:
  WorkspacePool(DeviceType device_type, std::shared_ptr<Device> device);
  ~WorkspacePool();

  void* AllocWorkspace(Context ctx, size_t size, double scale);
  void FreeWorkspace(Context ctx, void* ptr);
  size_t WorkspaceActualSize(Context, void* ptr);
  size_t TotalSize(Context ctx);
  size_t FreeSize(Context ctx);

 private:
  static constexpr int kMaxDevice = 32;

  class Pool;
  std::array<Pool*, kMaxDevice> _array;
  DeviceType _device_type;
  std::shared_ptr<Device> _device;
  std::mutex _mutex;
};

}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_WORKSPACE_POOL_H
