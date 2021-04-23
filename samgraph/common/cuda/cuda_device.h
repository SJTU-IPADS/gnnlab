#ifndef SAMGRAPH_GPU_DEVICE_H
#define SAMGRAPH_GPU_DEVICE_H

#include <cuda_runtime.h>

#include "../device.h"

namespace samgraph {
namespace common {
namespace cuda {

class GPUDevice final : public Device {
 public:
  void SetDevice(Context ctx) override;
  void *AllocDataSpace(Context ctx, size_t nbytes,
                       size_t alignment = kAllocAlignment) override;
  void FreeDataSpace(Context ctx, void *ptr) override;
  void *AllocWorkspace(Context ctx, size_t nbytes,
                       size_t scale = Config::kAllocScale) override;
  void FreeWorkspace(Context ctx, void *ptr, size_t nbytes = 0) override;
  void CopyDataFromTo(const void *from, size_t from_offset, void *to,
                      size_t to_offset, size_t nbytes, Context ctx_from,
                      Context ctx_to, StreamHandle stream) override;

  StreamHandle CreateStream(Context ctx) override;
  void FreeStream(Context ctx, StreamHandle stream) override;
  void StreamSync(Context ctx, StreamHandle stream) override;
  void SyncStreamFromTo(Context ctx, StreamHandle event_src,
                        StreamHandle event_dst) override;

  static const std::shared_ptr<GPUDevice> &Global();

 private:
  static void GPUCopy(const void *from, void *to, size_t nbytes,
                      cudaMemcpyKind kind, cudaStream_t stream);
  static void GPUCopyPeer(const void *from, int from_device, void *to,
                          int to_device, size_t nbytes, cudaStream_t stream);
};

}  // namespace cuda
}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_GPU_DEVICE_H
