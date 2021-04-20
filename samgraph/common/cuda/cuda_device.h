#ifndef SAMGRAPH_CUDA_DEVICE_H
#define SAMGRAPH_CUDA_DEVICE_H

#include <cuda_runtime.h>

#include "../device.h"

namespace samgraph {
namespace common {
namespace cuda {

class CudaDevice final : public Device {
 public:
  void SetDevice(Context ctx) override;
  void *AllocDataSpace(Context ctx, size_t nbytes, size_t alignment) override;
  void FreeDataSpace(Context ctx, void *ptr) override;
  void *AllocWorkspace(
      Context ctx, size_t nbytes,
      size_t scale_factor = Config::kAllocScaleFactor) override;
  void FreeWorkspace(Context ctx, void *ptr) override;
  void CopyDataFromTo(const void *from, size_t from_offset, void *to,
                      size_t to_offset, size_t nbytes, Context ctx_from,
                      Context ctx_to, StreamHandle stream) override;

  StreamHandle CreateStream(Context ctx) override;
  void FreeStream(Context ctx, StreamHandle stream) override;
  void StreamSync(Context ctx, StreamHandle stream) override;
  void SyncStreamFromTo(Context ctx, StreamHandle event_src,
                        StreamHandle event_dst) override;

  static const std::shared_ptr<CudaDevice> &Global();

 private:
  static void GpuCopy(const void *from, void *to, size_t size,
                      cudaMemcpyKind kind, cudaStream_t stream);
};

}  // namespace cuda
}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_CUDA_DEVICE_H
