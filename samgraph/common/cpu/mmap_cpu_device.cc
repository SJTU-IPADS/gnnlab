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

void MmapCpuDevice::SetDevice(Context ctx) {}

void *MmapCpuDevice::AllocDataSpace(Context ctx, size_t nbytes,
                                    size_t alignment) {
  LOG(FATAL) << "Device does not support AllocDataSpace api";
  return nullptr;
}

void MmapCpuDevice::FreeDataSpace(Context ctx, void *ptr) {}

void MmapCpuDevice::CopyDataFromTo(const void *from, size_t from_offset,
                                   void *to, size_t to_offset, size_t nbytes,
                                   Context ctx_from, Context ctx_to,
                                   StreamHandle stream) {
  LOG(FATAL) << "Device does not support CopyDataFromTo api";
}

void MmapCpuDevice::StreamSync(Context ctx, StreamHandle stream) {
  LOG(FATAL) << "Device does not support StreamSync api";
}

void *MmapCpuDevice::AllocWorkspace(Context ctx, size_t nbytes, size_t scale) {
  LOG(FATAL) << "Device does not support AllocWorkspace api";
  return nullptr;
}

void MmapCpuDevice::FreeWorkspace(Context ctx, void *data, size_t nbytes) {
  munmap(data, nbytes);
}

const std::shared_ptr<MmapCpuDevice> &MmapCpuDevice::Global() {
  static std::shared_ptr<MmapCpuDevice> inst =
      std::make_shared<MmapCpuDevice>();
  return inst;
}

}  // namespace cpu
}  // namespace common
}  // namespace samgraph
