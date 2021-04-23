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

void *MmapCPUDevice::AllocWorkspace(Context ctx, size_t nbytes, size_t scale) {
  LOG(FATAL) << "Device does not support AllocWorkspace api";
  return nullptr;
}

void MmapCPUDevice::FreeWorkspace(Context ctx, void *data, size_t nbytes) {
  munmap(data, nbytes);
}

const std::shared_ptr<MmapCPUDevice> &MmapCPUDevice::Global() {
  static std::shared_ptr<MmapCPUDevice> inst =
      std::make_shared<MmapCPUDevice>();
  return inst;
}

}  // namespace cpu
}  // namespace common
}  // namespace samgraph
