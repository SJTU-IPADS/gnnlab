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

#include "cpu_device.h"

#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>
#include <memory>

#include "../logging.h"
#include "../workspace_pool.h"
#include "../run_config.h"

namespace samgraph {
namespace common {
namespace cpu {

void CPUDevice::SetDevice(Context ctx) {}

void *CPUDevice::AllocDataSpace(Context ctx, size_t nbytes, size_t alignment) {
  void *ptr;
  if (ctx.device_id == CPU_CUDA_HOST_MALLOC_DEVICE) {
    CUDA_CALL(cudaHostAlloc(&ptr, nbytes, cudaHostAllocDefault));
  } else if (ctx.device_id == CPU_CLIB_MALLOC_DEVICE) {
    int ret = posix_memalign(&ptr, alignment, nbytes);
    CHECK_EQ(ret, 0);
  } else {
    CHECK(false);
  }

  return ptr;
}

void CPUDevice::FreeDataSpace(Context ctx, void *ptr) {
  if (ctx.device_id == CPU_CUDA_HOST_MALLOC_DEVICE) {
    CUDA_CALL(cudaFreeHost(ptr));
  } else if (ctx.device_id == CPU_CLIB_MALLOC_DEVICE) {
    free(ptr);
  } else {
    CHECK(false);
  }
}

void CPUDevice::CopyDataFromTo(const void *from, size_t from_offset, void *to,
                               size_t to_offset, size_t nbytes,
                               Context ctx_from, Context ctx_to,
                               StreamHandle stream) {
  // avoid copying with gpu
  CHECK(ctx_from.device_type != kGPU && ctx_to.device_type != kGPU);
  memcpy(static_cast<char *>(to) + to_offset,
         static_cast<const char *>(from) + from_offset, nbytes);
}

void CPUDevice::StreamSync(Context ctx, StreamHandle stream) {}

const std::shared_ptr<CPUDevice> &CPUDevice::Global() {
  static std::shared_ptr<CPUDevice> inst = std::make_shared<CPUDevice>();
  return inst;
}

struct CPUMemoryPool : public WorkspacePool {
  CPUMemoryPool() : WorkspacePool(kCPU, CPUDevice::Global()) {}
};

std::shared_ptr<WorkspacePool> &CPUWorkspacePool() {
  static std::shared_ptr<WorkspacePool> inst =
      std::make_shared<WorkspacePool>(kCPU, CPUDevice::Global());
  return inst;
}

void *CPUDevice::AllocWorkspace(Context ctx, size_t nbytes, double scale) {
  return CPUWorkspacePool()->AllocWorkspace(ctx, nbytes, scale);
}

void CPUDevice::FreeWorkspace(Context ctx, void *data, size_t nbytes) {
  CPUWorkspacePool()->FreeWorkspace(ctx, data);
}

size_t CPUDevice::WorkspaceActualSize(Context ctx, void *ptr) {
  return CPUWorkspacePool()->WorkspaceActualSize(ctx, ptr);
}

}  // namespace cpu
}  // namespace common
}  // namespace samgraph
