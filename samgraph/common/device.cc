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
        _device[type] = cpu::CPUDevice::Global().get();
        break;
      case kGPU:
      case kGPU_UM:
        _device[type] = cuda::GPUDevice::Global().get();
        break;
      case kMMAP:
        _device[type] = cpu::MmapCPUDevice::Global().get();
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

void* Device::AllocWorkspace(Context ctx, size_t nbytes, double scale) {
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
