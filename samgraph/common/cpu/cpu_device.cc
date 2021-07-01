#include "cpu_device.h"

#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>
#include <memory>

#include "../logging.h"
#include "../workspace_pool.h"

namespace samgraph {
namespace common {
namespace cpu {

void CPUDevice::SetDevice(Context ctx) {}

void *CPUDevice::AllocDataSpace(Context ctx, size_t nbytes, size_t alignment) {
  void *ptr;
  CUDA_CALL(cudaHostAlloc(&ptr, nbytes, cudaHostAllocDefault));

  // int ret = posix_memalign(&ptr, alignment, nbytes);
  // CHECK_EQ(ret, 0);

  return ptr;
}

void CPUDevice::FreeDataSpace(Context ctx, void *ptr) {
  CUDA_CALL(cudaFreeHost(ptr));

  // free(ptr);
}

void CPUDevice::CopyDataFromTo(const void *from, size_t from_offset, void *to,
                               size_t to_offset, size_t nbytes,
                               Context ctx_from, Context ctx_to,
                               StreamHandle stream) {
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

}  // namespace cpu
}  // namespace common
}  // namespace samgraph
