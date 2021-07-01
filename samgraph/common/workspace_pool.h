#ifndef SAMGRAPH_WORKSPACE_POOL_H
#define SAMGRAPH_WORKSPACE_POOL_H

#include <array>
#include <memory>
#include <mutex>

#include "common.h"
#include "device.h"

namespace samgraph {
namespace common {

class WorkspacePool {
 public:
  WorkspacePool(DeviceType device_type, std::shared_ptr<Device> device);
  ~WorkspacePool();

  void* AllocWorkspace(Context ctx, size_t size, double scale);
  void FreeWorkspace(Context ctx, void* ptr);

 private:
  static constexpr int kMaxDevice = 32;

  class Pool;
  std::array<Pool*, kMaxDevice> _array;
  DeviceType _device_type;
  std::shared_ptr<Device> _device;
  std::mutex _mutex;
};

}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_WORKSPACE_POOL_H
