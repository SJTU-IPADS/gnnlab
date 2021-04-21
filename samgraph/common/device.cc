#include "device.h"

#include <array>
#include <mutex>

#include "cpu/cpu_device.h"
#include "cpu/mmap_cpu_device.h"
#include "cuda/cuda_device.h"
#include "logging.h"

namespace samgraph {
namespace common {

class DeviceManager {
 public:
  static constexpr int kMaxDevice = 32;
  static Device* Get(const Context& ctx) { return Get(ctx.device_type); }
  static Device* Get(int dev_type) { return Global()->GetDevice(dev_type); }

 private:
  std::array<Device*, kMaxDevice> _device;
  std::mutex _mutex;

  DeviceManager() { std::fill(_device.begin(), _device.end(), nullptr); }

  // Global static variable.
  static DeviceManager* Global() {
    static DeviceManager inst;
    return &inst;
  }

  // Get or initialize Device.
  Device* GetDevice(int type) {
    if (_device[type] != nullptr) return _device[type];
    std::lock_guard<std::mutex> lock(_mutex);
    if (_device[type] != nullptr) return _device[type];
    switch (type) {
      case kCPU:
        _device[type] = cpu::CpuDevice::Global().get();
        break;
      case kGPU:
        _device[type] = cuda::GpuDevice::Global().get();
        break;
      case kMMAP:
        _device[type] = cpu::MmapCpuDevice::Global().get();
        break;
      default:
        CHECK(0);
    }
    return _device[type];
  }
};

Device* Device::Get(Context ctx) {
  return DeviceManager::Get(static_cast<int>(ctx.device_type));
}

void* Device::AllocWorkspace(Context ctx, size_t nbytes, size_t scale) {
  return AllocDataSpace(ctx, nbytes, kTempAllocaAlignment);
}

void Device::FreeWorkspace(Context ctx, void* ptr, size_t nbytes) {
  FreeDataSpace(ctx, ptr);
}

StreamHandle Device::CreateStream(Context ctx) {
  LOG(FATAL) << "Device does not support stream api.";
  return nullptr;
}

void Device::FreeStream(Context ctx, StreamHandle stream) {
  LOG(FATAL) << "Device does not support stream api.";
}

void Device::SyncStreamFromTo(Context ctx, StreamHandle event_src,
                              StreamHandle event_dst) {
  LOG(FATAL) << "Device does not support stream api.";
}

}  // namespace common
}  // namespace samgraph
