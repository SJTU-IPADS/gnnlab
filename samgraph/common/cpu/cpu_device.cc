#include "cpu_device.h"

#include <cstdlib>
#include <cstring>
#include <memory>

#include "../logging.h"
#include "../thread_local.h"
#include "../workspace_pool.h"

namespace samgraph {
namespace common {
namespace cpu {

void CpuDevice::SetDevice(Context ctx) {}

void *CpuDevice::AllocDataSpace(Context ctx, size_t nbytes, size_t alignment) {
  void *ptr;
  int ret = posix_memalign(&ptr, alignment, nbytes);
  CHECK_EQ(ret, 0);

  return ptr;
}

void CpuDevice::FreeDataSpace(Context ctx, void *ptr) { free(ptr); }

void CpuDevice::CopyDataFromTo(const void *from, size_t from_offset, void *to,
                               size_t to_offset, size_t nbytes,
                               Context ctx_from, Context ctx_to,
                               StreamHandle stream) {
  memcpy(static_cast<char *>(to) + to_offset,
         static_cast<const char *>(from) + from_offset, nbytes);
}

void CpuDevice::StreamSync(Context ctx, StreamHandle stream) {}

const std::shared_ptr<CpuDevice> &CpuDevice::Global() {
  static std::shared_ptr<CpuDevice> inst = std::make_shared<CpuDevice>();
  return inst;
}

struct CpuMemoryPool : public WorkspacePool {
  CpuMemoryPool() : WorkspacePool(kCPU, CpuDevice::Global()) {}
};

std::shared_ptr<WorkspacePool> &CpuWorkspacePool() {
  static std::shared_ptr<WorkspacePool> inst =
      std::make_shared<WorkspacePool>(kCPU, CpuDevice::Global());
  return inst;
}

void *CpuDevice::AllocWorkspace(Context ctx, size_t nbytes, size_t scale) {
  return CpuWorkspacePool()->AllocWorkspace(ctx, nbytes, scale);
}

void CpuDevice::FreeWorkspace(Context ctx, void *data, size_t nbytes) {
  CpuWorkspacePool()->FreeWorkspace(ctx, data);
}

}  // namespace cpu
}  // namespace common
}  // namespace samgraph
