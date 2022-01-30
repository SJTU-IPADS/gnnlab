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

#include "mmap_cpu_device.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <memory>
#include <unordered_map>

#include "../logging.h"

namespace samgraph {
namespace common {
namespace cpu {

void MmapCPUDevice::SetDevice(Context ctx) {}

void *MmapCPUDevice::AllocDataSpace(Context ctx, size_t nbytes,
                                    size_t alignment) {
  LOG(FATAL) << "Device does not support AllocDataSpace api";
  return nullptr;
}

void MmapCPUDevice::FreeDataSpace(Context ctx, void *ptr) {}

void MmapCPUDevice::CopyDataFromTo(const void *from, size_t from_offset,
                                   void *to, size_t to_offset, size_t nbytes,
                                   Context ctx_from, Context ctx_to,
                                   StreamHandle stream) {
  LOG(FATAL) << "Device does not support CopyDataFromTo api";
}

void MmapCPUDevice::StreamSync(Context ctx, StreamHandle stream) {
  LOG(FATAL) << "Device does not support StreamSync api";
}

void *MmapCPUDevice::AllocWorkspace(Context ctx, size_t nbytes, double scale) {
  LOG(FATAL) << "Device does not support AllocWorkspace api";
  return nullptr;
}

void MmapCPUDevice::FreeWorkspace(Context ctx, void *data, size_t nbytes) {
  int ret = munmap(data, nbytes);
  CHECK_EQ(ret, 0);
}

const std::shared_ptr<MmapCPUDevice> &MmapCPUDevice::Global() {
  static std::shared_ptr<MmapCPUDevice> inst =
      std::make_shared<MmapCPUDevice>();
  return inst;
}

}  // namespace cpu
}  // namespace common
}  // namespace samgraph
