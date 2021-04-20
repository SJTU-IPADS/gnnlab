#ifndef SAMGRAPH_WORKSPACE_POOL_H
#define SAMGRAPH_WORKSPACE_POOL_H

#include <memory>
#include <vector>

#include "common.h"
#include "device.h"

namespace samgraph {
namespace common {

class WorkspacePool {
 public:
  WorkspacePool(DeviceType device_type, std::shared_ptr<Device> device);
  ~WorkspacePool();

  void* AllocWorkspace(Context ctx, size_t size, size_t scale_factor);
  void FreeWorkspace(Context ctx, void* ptr);

 private:
  class Pool;
  std::vector<Pool*> _array;
  DeviceType _device_type;
  std::shared_ptr<Device> _device;
};

}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_WORKSPACE_POOL_H
