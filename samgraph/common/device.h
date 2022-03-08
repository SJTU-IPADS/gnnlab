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

#ifndef SAMGRAPH_DEVICE_H
#define SAMGRAPH_DEVICE_H

#include <cstdint>
#include <array>

#include "common.h"
#include "constant.h"

namespace samgraph {
namespace common {

// Number of bytes each allocation must align to
constexpr int kAllocAlignment = 64;

// Number of bytes each allocation must align to in temporary allocation
constexpr int kTempAllocaAlignment = 64;

class Device {
 public:
  virtual ~Device() {}
  virtual void SetDevice(Context ctx) = 0;
  virtual void *AllocDataSpace(Context ctx, size_t nbytes,
                               size_t alignment = kAllocAlignment) = 0;
  virtual void FreeDataSpace(Context ctx, void *ptr) = 0;
  virtual void *AllocWorkspace(Context ctx, size_t nbytes,
                               double scale = Constant::kAllocScale);
  virtual void FreeWorkspace(Context ctx, void *ptr, size_t nbytes = 0);
  virtual void CopyDataFromTo(const void *from, size_t from_offset, void *to,
                              size_t to_offset, size_t nbytes, Context ctx_from,
                              Context ctx_to, StreamHandle stream = 0) = 0;

  virtual StreamHandle CreateStream(Context ctx);
  virtual void FreeStream(Context ctx, StreamHandle stream);
  virtual void StreamSync(Context ctx, StreamHandle stream) = 0;
  virtual void SyncDevice(Context ctx) {}
  virtual void SyncStreamFromTo(Context ctx, StreamHandle event_src,
                                StreamHandle event_dst);
  virtual size_t TotalSize(Context ctx) {return 0;};
  virtual size_t DataSize(Context ctx) {return 0;};
  virtual size_t WorkspaceSize(Context ctx) {return 0;};
  virtual size_t FreeWorkspaceSize(Context ctx) {return 0;};
  static Device *Get(Context ctx);
};
}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_DEVICE_H
