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

#include "cuda_device.h"

#include <cuda_runtime.h>

#include "../logging.h"
#include "../workspace_pool.h"
#include "cuda_common.h"
#include "../run_config.h"

namespace samgraph {
namespace common {
namespace cuda {

GPUDevice::GPUDevice() {
  _allocated_size_list = new size_t[32]();
  for (int i = 0; i < 32; i++) {
    _allocated_size_list[i] = 0;
  }
}

void GPUDevice::SetDevice(Context ctx) {
  CUDA_CALL(cudaSetDevice(ctx.device_id));
}

void *GPUDevice::AllocDataSpace(Context ctx, size_t nbytes, size_t alignment) {
  void *ret;
  CHECK_EQ(256 % alignment, 0U);
  CUDA_CALL(cudaSetDevice(ctx.device_id));
  if (ctx.device_type == kGPU) {
    CUDA_CALL(cudaMalloc(&ret, nbytes));
  } else if (ctx.device_type == kGPU_UM) {
    LOG(INFO) << "alloc unified memory " << ToReadableSize(nbytes);
    CUDA_CALL(cudaMallocManaged(&ret, nbytes));
    // advice gpu um
    LOG(INFO) << "use 2 device store graph!";
    CHECK(RunConfig::unified_memory_ctxes.size() >= 2);
    auto ctx0 = RunConfig::unified_memory_ctxes[0];
    auto ctx1 = RunConfig::unified_memory_ctxes[1];
    size_t ctx0_nbytes = static_cast<size_t>(1.0 * nbytes * RunConfig::unified_memory_percentage);
    // round to page
    // ctx0_nbytes = ((ctx0_nbytes + 4096 - 1) / (4096)) * 4096;
    // ctx0_nbytes = std::min(ctx0_nbytes, nbytes);
    size_t ctx1_nbytes = nbytes - ctx0_nbytes;
    LOG(INFO) << "unified_memory: in " << ctx0  << " " << ToReadableSize(ctx0_nbytes)
              << ", in " << ctx1 << " " << ToReadableSize(ctx1_nbytes);
    if (ctx0_nbytes != 0) {
        CUDA_CALL(cudaMemAdvise(ret, ctx0_nbytes,
            cudaMemAdviseSetPreferredLocation, ctx0.GetCudaDeviceId()));
        CUDA_CALL(cudaMemAdvise(ret, ctx0_nbytes,
            cudaMemAdviseSetAccessedBy, ctx0.GetCudaDeviceId()));
    }
    if (ctx1_nbytes != 0) {
        CUDA_CALL(cudaMemAdvise(ret + ctx0_nbytes, ctx1_nbytes,
            cudaMemAdviseSetPreferredLocation, ctx1.GetCudaDeviceId()));
        CUDA_CALL(cudaMemAdvise(ret + ctx0_nbytes, ctx1_nbytes,
            cudaMemAdviseSetAccessedBy, ctx0.GetCudaDeviceId()));
    }
  } else if (ctx.device_type == DeviceType::kGPU_P2P) {
    LOG(INFO) << "sampler" << RunConfig::sampler_ctx.device_id << " accessp2p from " << ctx.device_id;
    CUDA_CALL(cudaMalloc(&ret, nbytes));
    CUDA_CALL(cudaSetDevice(RunConfig::sampler_ctx.device_id));
  } else {
      LOG(FATAL) << "device_type is not supported";
  }
  // data space is only allocated during init phase, thread-safe
  _allocated_size_list[ctx.device_id] += nbytes;
  return ret;
}

void GPUDevice::FreeDataSpace(Context ctx, void *ptr) {
  CUDA_CALL(cudaSetDevice(ctx.device_id));
  CUDA_CALL(cudaFree(ptr));
  /** data space is only allocated during init phase
    * freed workspace does not return here */
}

void GPUDevice::CopyDataFromTo(const void *from, size_t from_offset, void *to,
                               size_t to_offset, size_t nbytes,
                               Context ctx_from, Context ctx_to,
                               StreamHandle stream) {
  if (nbytes == 0) return;
  cudaStream_t cu_stream = static_cast<cudaStream_t>(stream);
  from = static_cast<const char *>(from) + from_offset;
  to = static_cast<char *>(to) + to_offset;
  if(ctx_from.device_type == DeviceType::kGPU_UM || ctx_from.device_type == DeviceType::kGPU_P2P) {
    ctx_from.device_type = DeviceType::kGPU;
  }
  if(ctx_to.device_type == DeviceType::kGPU_UM || ctx_to.device_type == DeviceType::kGPU_P2P) {
    ctx_to.device_type = DeviceType::kGPU;
  }
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
  } else if (ctx_from.device_type == kMMAP && ctx_to.device_type == kGPU) {
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

void GPUDevice::SyncDevice(Context ctx) {
  CUDA_CALL(cudaSetDevice(ctx.device_id));
  CUDA_CALL(cudaDeviceSynchronize());
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
  /** FIXME: no mechanism to ensure stream belongs to this ctx */
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
    cudaMemcpyPeerAsync(to, to_device, from, from_device, nbytes, stream);
  } else {
    cudaMemcpyPeer(to, to_device, from, from_device, nbytes);
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
size_t GPUDevice::WorkspaceActualSize(Context ctx, void *ptr) {
  return GPUWorkspacePool()->WorkspaceActualSize(ctx, ptr);
}
size_t GPUDevice::TotalSize(Context ctx) {
  return _allocated_size_list[ctx.device_id];
}
size_t GPUDevice::WorkspaceSize(Context ctx) {
  return GPUWorkspacePool()->TotalSize(ctx);
}
size_t GPUDevice::DataSize(Context ctx) {
  return TotalSize(ctx) - WorkspaceSize(ctx);
}
size_t GPUDevice::FreeWorkspaceSize(Context ctx) {
  return GPUWorkspacePool()->FreeSize(ctx);
}

}  // namespace cuda
}  // namespace common
}  // namespace samgraph
