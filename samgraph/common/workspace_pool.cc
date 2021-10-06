#include "workspace_pool.h"

#include <memory>
#include <unordered_map>

#include "device.h"
#include "logging.h"

namespace samgraph {
namespace common {

// page size.
constexpr size_t kWorkspacePageSize = 4 << 10;

class WorkspacePool::Pool {
 public:
  Pool() {
    // List gurad
    Entry e;
    e.data = nullptr;
    e.size = 0;

    _free_list.reserve(kListSize);
    _allocated.reserve(kListSize);

    _free_list.push_back(e);
    _allocated.push_back(e);
  }

  // allocate from pool
  void *Alloc(Context ctx, Device *device, size_t nbytes, double scale) {
    // Allocate align to page.
    std::lock_guard<std::mutex> lock(_mutex);
    nbytes = (nbytes + (kWorkspacePageSize - 1)) / kWorkspacePageSize *
             kWorkspacePageSize;
    if (nbytes == 0) nbytes = kWorkspacePageSize;

    Entry e;
    if (_free_list.size() == 1) {
      nbytes *= scale;
      e.data = device->AllocDataSpace(ctx, nbytes, kTempAllocaAlignment);
      e.size = nbytes;
    } else {
      if (_free_list.back().size >= nbytes) {
        // find smallest fit
        auto it = _free_list.end() - 2;
        for (; it->size >= nbytes; --it) {
        }
        e = *(it + 1);
        if (e.size > 2 * nbytes) {
          nbytes *= scale;
          e.data = device->AllocDataSpace(ctx, nbytes, kTempAllocaAlignment);
          e.size = nbytes;
        } else {
          _free_list.erase(it + 1);
          _free_list_total_size -= e.size;
        }
      } else {
        nbytes *= scale;
        e.data = device->AllocDataSpace(ctx, nbytes, kTempAllocaAlignment);
        e.size = nbytes;
      }
    }
    _allocated.push_back(e);
    _allocated_total_size += e.size;
    return e.data;
  }

  // free resource back to pool
  void Free(void *data) {
    std::lock_guard<std::mutex> lock(_mutex);
    Entry e;
    if (_allocated.back().data == data) {
      // quick path, last allocated.
      e = _allocated.back();
      _allocated.pop_back();
    } else {
      int index = static_cast<int>(_allocated.size()) - 2;
      for (; index > 0 && _allocated[index].data != data; --index) {
      }
      CHECK_GT(index, 0) << "trying to free things that has not been allocated";
      e = _allocated[index];
      _allocated.erase(_allocated.begin() + index);
    }
    _allocated_total_size -= e.size;

    if (_free_list.back().size < e.size) {
      _free_list.push_back(e);
    } else {
      size_t i = _free_list.size() - 1;
      _free_list.resize(_free_list.size() + 1);
      for (; e.size < _free_list[i].size; --i) {
        _free_list[i + 1] = _free_list[i];
      }
      _free_list[i + 1] = e;
    }
    _free_list_total_size += e.size;
  }

  // Release all resources
  void Release(Context ctx, Device *device) {
    CHECK_EQ(_allocated.size(), 1);
    for (size_t i = 1; i < _free_list.size(); ++i) {
      device->FreeDataSpace(ctx, _free_list[i].data);
    }
    _free_list.clear();
  }
  size_t TotalSize() {
    std::lock_guard<std::mutex> lock(_mutex);
    return _free_list_total_size + _allocated_total_size;
  }
  size_t FreeSize() {
    std::lock_guard<std::mutex> lock(_mutex);
    return _free_list_total_size;
  }

 private:
  /*! \brief a single entry in the pool */
  struct Entry {
    void *data;
    size_t size;
  };

  std::vector<Entry> _free_list;
  std::vector<Entry> _allocated;
  size_t _free_list_total_size = 0, _allocated_total_size = 0;
  std::mutex _mutex;

  constexpr static size_t kListSize = 100;
};

WorkspacePool::WorkspacePool(DeviceType device_type,
                             std::shared_ptr<Device> device)
    : _device_type(device_type), _device(device) {
  std::fill(_array.begin(), _array.end(), nullptr);
}

WorkspacePool::~WorkspacePool() {
  // for (size_t i = 0; i < _array.size(); ++i) {
  //   if (_array[i] != nullptr) {
  //     Context ctx;
  //     ctx.device_type = _device_type;
  //     ctx.device_id = static_cast<int>(i);
  //     _array[i]->Release(ctx, _device.get());
  //     delete _array[i];
  //   }
  // }
}

void *WorkspacePool::AllocWorkspace(Context ctx, size_t size, double scale) {
  if (_array[ctx.device_id] != nullptr) {
    return _array[ctx.device_id]->Alloc(ctx, _device.get(), size, scale);
  }

  std::lock_guard<std::mutex> lock(_mutex);
  if (_array[ctx.device_id] == nullptr) {
    _array[ctx.device_id] = new Pool();
  }

  return _array[ctx.device_id]->Alloc(ctx, _device.get(), size, scale);
}

void WorkspacePool::FreeWorkspace(Context ctx, void *ptr) {
  CHECK(static_cast<size_t>(ctx.device_id) < _array.size() &&
        _array[ctx.device_id] != nullptr);
  _array[ctx.device_id]->Free(ptr);
}

size_t WorkspacePool::TotalSize(Context ctx) {
  if (_array[ctx.device_id] == nullptr) return 0;
  return _array[ctx.device_id]->TotalSize();
}
size_t WorkspacePool::FreeSize(Context ctx) {
  if (_array[ctx.device_id] == nullptr) return 0;
  return _array[ctx.device_id]->FreeSize();
}

}  // namespace common
}  // namespace samgraph
