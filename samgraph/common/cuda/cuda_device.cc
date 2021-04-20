#include "cuda_device.h"

#include <cuda_runtime.h>

#include "../logging.h"
#include "../thread_local.h"
#include "../workspace_pool.h"
#include "cuda_common.h"

namespace samgraph {
namespace common {
namespace cuda {

void CudaDevice::SetDevice(Context ctx) {
  CUDA_CALL(cudaSetDevice(ctx.device_id));
}

void *CudaDevice::AllocDataSpace(Context ctx, size_t nbytes, size_t alignment) {
  void *ret;
  CHECK_EQ(256 % alignment, 0U);
  CUDA_CALL(cudaSetDevice(ctx.device_id));
  CUDA_CALL(cudaMalloc(&ret, nbytes));
  return ret;
}

void CudaDevice::FreeDataSpace(Context ctx, void *ptr) {
  CUDA_CALL(cudaSetDevice(ctx.device_id));
  CUDA_CALL(cudaFree(ptr));
}

void CudaDevice::CopyDataFromTo(const void *from, size_t from_offset, void *to,
                                size_t to_offset, size_t nbytes,
                                Context ctx_from, Context ctx_to,
                                StreamHandle stream) {
  cudaStream_t cu_stream = static_cast<cudaStream_t>(stream);
  from = static_cast<const char *>(from) + from_offset;
  to = static_cast<char *>(to) + to_offset;
  if (ctx_from.device_type == kGPU && ctx_to.device_type == kGPU) {
    CUDA_CALL(cudaSetDevice(ctx_from.device_id));
    if (ctx_from.device_id == ctx_to.device_id) {
      GpuCopy(from, to, nbytes, cudaMemcpyDeviceToDevice, cu_stream);
    } else {
      cudaMemcpyPeerAsync(to, ctx_to.device_id, from, ctx_from.device_id,
                          nbytes, cu_stream);
    }
  } else if (ctx_from.device_type == kGPU && ctx_to.device_type == kCPU) {
    CUDA_CALL(cudaSetDevice(ctx_from.device_id));
    GpuCopy(from, to, nbytes, cudaMemcpyDeviceToHost, cu_stream);
  } else if (ctx_from.device_type == kCPU && ctx_to.device_type == kGPU) {
    CUDA_CALL(cudaSetDevice(ctx_to.device_id));
    GpuCopy(from, to, nbytes, cudaMemcpyHostToDevice, cu_stream);
  } else {
    LOG(FATAL) << "expect copy from/to GPU or between GPU";
  }
}

const std::shared_ptr<CudaDevice> &CudaDevice::Global() {
  static std::shared_ptr<CudaDevice> inst = std::make_shared<CudaDevice>();
  return inst;
}

StreamHandle CudaDevice::CreateStream(Context ctx) {
  CUDA_CALL(cudaSetDevice(ctx.device_id));
  cudaStream_t retval;
  CUDA_CALL(cudaStreamCreateWithFlags(&retval, cudaStreamNonBlocking));
  return static_cast<StreamHandle>(retval);
}

void CudaDevice::FreeStream(Context ctx, StreamHandle stream) {
  CUDA_CALL(cudaSetDevice(ctx.device_id));
  cudaStream_t cu_stream = static_cast<cudaStream_t>(stream);
  CUDA_CALL(cudaStreamDestroy(cu_stream));
}

void CudaDevice::SyncStreamFromTo(Context ctx, StreamHandle event_src,
                                  StreamHandle event_dst) {
  CUDA_CALL(cudaSetDevice(ctx.device_id));
  cudaStream_t src_stream = static_cast<cudaStream_t>(event_src);
  cudaStream_t dst_stream = static_cast<cudaStream_t>(event_dst);
  cudaEvent_t evt;
  CUDA_CALL(cudaEventCreate(&evt));
  CUDA_CALL(cudaEventRecord(evt, src_stream));
  CUDA_CALL(cudaStreamWaitEvent(dst_stream, evt, 0));
  CUDA_CALL(cudaEventDestroy(evt));
}

void CudaDevice::StreamSync(Context ctx, StreamHandle stream) {
  CUDA_CALL(cudaSetDevice(ctx.device_id));
  CUDA_CALL(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
}

void CudaDevice::GpuCopy(const void *from, void *to, size_t size,
                         cudaMemcpyKind kind, cudaStream_t stream) {
  if (stream != 0) {
    CUDA_CALL(cudaMemcpyAsync(to, from, size, kind, stream));
  } else {
    CUDA_CALL(cudaMemcpy(to, from, size, kind));
  }
}

struct CudaMemoryPool : public WorkspacePool {
  CudaMemoryPool() : WorkspacePool(kCPU, CudaDevice::Global()) {}
};

void *CudaDevice::AllocWorkspace(Context ctx, size_t nbytes,
                                 size_t scale_factor) {
  return ThreadLocalStore<CudaMemoryPool>::Get()->AllocWorkspace(ctx, nbytes,
                                                                 scale_factor);
}

void CudaDevice::FreeWorkspace(Context ctx, void *data) {
  ThreadLocalStore<CudaMemoryPool>::Get()->FreeWorkspace(ctx, data);
}

}  // namespace cuda
}  // namespace common
}  // namespace samgraph
