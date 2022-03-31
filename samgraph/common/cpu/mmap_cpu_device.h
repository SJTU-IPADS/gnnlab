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

#ifndef SAMGRAPH_MMAP_DEVICE_H
#define SAMGRAPH_MMAP_DEVICE_H

#include "../device.h"

namespace samgraph {
namespace common {
namespace cpu {

class MmapCPUDevice final : public Device {
 public:
  void SetDevice(Context ctx) override;
  void *AllocDataSpace(Context ctx, size_t nbytes, size_t alignment) override;
  void FreeDataSpace(Context ctx, void *ptr) override;
  void *AllocWorkspace(Context ctx, size_t nbytes,
                       double scale = Constant::kAllocScale) override;
  void FreeWorkspace(Context ctx, void *ptr, size_t nbytes = 0) override;
  void CopyDataFromTo(const void *from, size_t from_offset, void *to,
                      size_t to_offset, size_t nbytes, Context ctx_from,
                      Context ctx_to, StreamHandle stream) override;

  void StreamSync(Context ctx, StreamHandle stream) override;

  static const std::shared_ptr<MmapCPUDevice> &Global();
};

}  // namespace cpu
}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_MMAP_DEVICE_H
