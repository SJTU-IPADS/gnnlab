#include "cuda_device.h"

#include <cuda_runtime.h>

#include "../logging.h"
#include "../workspace_pool.h"
#include "cuda_common.h"

namespace samgraph {
namespace common {
namespace cuda {

void GPUDevice::SetDevice(Context ctx) {
  CUDA_CALL(cudaSetDevice(ctx.device_id));
}

void *GPUDevice::AllocDataSpace(Context ctx, size_t nbytes, size_t alignment) {
  void *ret;
  CHECK_EQ(256 % alignment, 0U);
  CUDA_CALL(cudaSetDevice(ctx.device_id));
  CUDA_CALL(cudaMalloc(&ret, nbytes));
  return ret;
}

void GPUDevice::FreeDataSpace(Context ctx, void *ptr) {
  CUDA_CALL(cudaSetDevice(ctx.device_id));
  CUDA_CALL(cudaFree(ptr));
}

void GPUDevice::CopyDataFromTo(const void *from, size_t from_offset, void *to,
                               size_t to_offset, size_t nbytes,
                               Context ctx_from, Context ctx_to,
                               StreamHandle stream) {
  cudaStream_t cu_stream = static_cast<cudaStream_t>(stream);
  from = static_cast<const char *>(from) + from_offset;
  to = static_cast<char *>(to) + to_offset;
  if (ctx_from.device_type == kGPU && ctx_to.device_type == kGPU) {
    CUDA_CALL(cudaSetDevice(ctx_from.device_id));
    if (ctx_from.device_id == ctx_to.device_id) {
      GPUCopy(from, to, nbytes, cudaMemcpyDeviceToDevice, cu_stream);
    } else {
      GPUCopyPeer(from, ctx_from.device_id, to, ctx_to.device_id, nbytes,
                  cu_stream);
    }
  } else if (ctx_from.device_type == kGPU && ctx_to.device_type == kCPU) {
    CUDA_CALL(cudaSetDevice(ctx_from.device_id));
    GPUCopy(from, to, nbytes, cudaMemcpyDeviceToHost, cu_stream);
  } else if (ctx_from.device_type == kCPU && ctx_to.device_type == kGPU) {
    CUDA_CALL(cudaSetDevice(ctx_to.device_id));
    GPUCopy(from, to, nbytes, cudaMemcpyHostToDevice, cu_stream);
  } else {
    LOG(FATAL) << "expect copy from/to GPU or between GPU";
  }
}

const std::shared_ptr<GPUDevice> &GPUDevice::Global() {
  static std::shared_ptr<GPUDevice> inst = std::make_shared<GPUDevice>();
  return inst;
}

StreamHandle GPUDevice::CreateStream(Context ctx) {
  CUDA_CALL(cudaSetDevice(ctx.device_id));
  cudaStream_t retval;
  CUDA_CALL(cudaStreamCreateWithFlags(&retval, cudaStreamNonBlocking));
  return static_cast<StreamHandle>(retval);
}

void GPUDevice::FreeStream(Context ctx, StreamHandle stream) {
  CUDA_CALL(cudaSetDevice(ctx.device_id));
  cudaStream_t cu_stream = static_cast<cudaStream_t>(stream);
  CUDA_CALL(cudaStreamDestroy(cu_stream));
}

void GPUDevice::SyncStreamFromTo(Context ctx, StreamHandle event_src,
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

void GPUDevice::StreamSync(Context ctx, StreamHandle stream) {
  if (stream != 0) {
    CUDA_CALL(cudaSetDevice(ctx.device_id));
    CUDA_CALL(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
  }
}

void GPUDevice::GPUCopy(const void *from, void *to, size_t nbytes,
                        cudaMemcpyKind kind, cudaStream_t stream) {
  if (stream != 0) {
    CUDA_CALL(cudaMemcpyAsync(to, from, nbytes, kind, stream));
  } else {
    CUDA_CALL(cudaMemcpy(to, from, nbytes, kind));
  }
}

void GPUDevice::GPUCopyPeer(const void *from, int from_device, void *to,
                            int to_device, size_t nbytes, cudaStream_t stream) {
  if (stream != 0) {
    cudaMemcpyPeerAsync(to, from_device, from, to_device, nbytes, stream);
  } else {
    cudaMemcpyPeer(to, from_device, from, to_device, nbytes);
  }
}

std::shared_ptr<WorkspacePool> &GPUWorkspacePool() {
  static std::shared_ptr<WorkspacePool> inst =
      std::make_shared<WorkspacePool>(kCPU, GPUDevice::Global());
  return inst;
}

void *GPUDevice::AllocWorkspace(Context ctx, size_t nbytes, double scale) {
  CUDA_CALL(cudaSetDevice(ctx.device_id));
  return GPUWorkspacePool()->AllocWorkspace(ctx, nbytes, scale);
}

void GPUDevice::FreeWorkspace(Context ctx, void *data, size_t nbytes) {
  CUDA_CALL(cudaSetDevice(ctx.device_id));
  GPUWorkspacePool()->FreeWorkspace(ctx, data);
}

}  // namespace cuda
}  // namespace common
}  // namespace samgraph
