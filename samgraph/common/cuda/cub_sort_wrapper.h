#pragma once
#include "../common.h"
#include "../device.h"
#include "../timer.h"
#include "../logging.h"
#include <cub/cub.cuh>
namespace samgraph{
namespace common{
namespace cuda{

template<typename NativeKey_t>
void CubSortKey(
    NativeKey_t* & key, NativeKey_t* & key_alter,
    const size_t len, Context gpu_ctx, 
    int begin_bit = 0, int end_bit = sizeof(NativeKey_t) * 8,
    StreamHandle stream = nullptr) {
  auto cu_stream = static_cast<cudaStream_t>(stream);
  auto device = Device::Get(gpu_ctx);

  cub::DoubleBuffer<NativeKey_t> keys(key, key_alter);

  size_t workspace_bytes;
  void * workspace = nullptr;
  CUDA_CALL(cub::DeviceRadixSort::SortKeys(
      workspace, workspace_bytes, keys, len,
      begin_bit, end_bit, cu_stream));

  workspace = device->AllocWorkspace(gpu_ctx, workspace_bytes);
  CUDA_CALL(cub::DeviceRadixSort::SortKeys(
      workspace, workspace_bytes, keys, len,
      begin_bit, end_bit, cu_stream));
  device->StreamSync(gpu_ctx, stream);

  key = keys.Current();
  key_alter = keys.Alternate();

  device->FreeWorkspace(gpu_ctx, workspace);
}

template<typename NativeKey_t>
void CubSortKey(
    const NativeKey_t* key_in, NativeKey_t* key_out,
    const size_t num_nodes, Context gpu_ctx,
    int begin_bit = 0, int end_bit = sizeof(NativeKey_t) * 8,
    StreamHandle stream=nullptr) {
  auto cu_stream = static_cast<cudaStream_t>(stream);
  auto device = Device::Get(gpu_ctx);

  size_t workspace_bytes;
  void *workspace = nullptr;
  CUDA_CALL(cub::DeviceRadixSort::SortKeys(
      workspace, workspace_bytes, key_in, key_out, num_nodes,
      begin_bit, end_bit, cu_stream));

  workspace = device->AllocWorkspace(gpu_ctx, workspace_bytes);
  CUDA_CALL(cub::DeviceRadixSort::SortKeys(
      workspace, workspace_bytes, key_in, key_out, num_nodes,
      begin_bit, end_bit, cu_stream));
  device->StreamSync(gpu_ctx, stream);

  device->FreeWorkspace(gpu_ctx, workspace);
}

template<typename NativeKey_t>
void CubSortKeyDescending(
    NativeKey_t* & key, NativeKey_t* & key_alter,
    const size_t len, Context gpu_ctx, 
    int begin_bit = 0, int end_bit = sizeof(NativeKey_t) * 8,
    StreamHandle stream = nullptr) {
  auto cu_stream = static_cast<cudaStream_t>(stream);
  auto device = Device::Get(gpu_ctx);

  cub::DoubleBuffer<NativeKey_t> keys(key, key_alter);

  size_t workspace_bytes;
  void * workspace = nullptr;
  CUDA_CALL(cub::DeviceRadixSort::SortKeysDescending(
      workspace, workspace_bytes, keys, len,
      begin_bit, end_bit, cu_stream));

  workspace = device->AllocWorkspace(gpu_ctx, workspace_bytes);
  CUDA_CALL(cub::DeviceRadixSort::SortKeysDescending(
      workspace, workspace_bytes, keys, len,
      begin_bit, end_bit, cu_stream));
  device->StreamSync(gpu_ctx, stream);

  key = keys.Current();
  key_alter = keys.Alternate();

  device->FreeWorkspace(gpu_ctx, workspace);
}

template<typename NativeKey_t>
void CubSortKeyDescending(
    const NativeKey_t* key_in, NativeKey_t* key_out,
    const size_t num_nodes, Context gpu_ctx,
    int begin_bit = 0, int end_bit = sizeof(NativeKey_t) * 8,
    StreamHandle stream=nullptr) {
  auto cu_stream = static_cast<cudaStream_t>(stream);
  auto device = Device::Get(gpu_ctx);

  size_t workspace_bytes;
  void *workspace = nullptr;
  CUDA_CALL(cub::DeviceRadixSort::SortKeysDescending(
      workspace, workspace_bytes, key_in, key_out, num_nodes,
      begin_bit, end_bit, cu_stream));

  workspace = device->AllocWorkspace(gpu_ctx, workspace_bytes);
  CUDA_CALL(cub::DeviceRadixSort::SortKeysDescending(
      workspace, workspace_bytes, key_in, key_out, num_nodes,
      begin_bit, end_bit, cu_stream));
  device->StreamSync(gpu_ctx, stream);

  device->FreeWorkspace(gpu_ctx, workspace);
}

template<typename NativeKey_t, typename NativeVal_t>
void CubSortPair(
    NativeKey_t* & key, NativeKey_t* & key_alter,
    NativeVal_t* & val, NativeVal_t* & val_alter,
    const size_t len, Context gpu_ctx, 
    int begin_bit = 0, int end_bit = sizeof(NativeKey_t) * 8,
    StreamHandle stream = nullptr) {
  auto cu_stream = static_cast<cudaStream_t>(stream);
  auto device = Device::Get(gpu_ctx);

  cub::DoubleBuffer<NativeKey_t> keys(key, key_alter);
  cub::DoubleBuffer<NativeVal_t> vals(val, val_alter);

  size_t workspace_bytes;
  void * workspace = nullptr;
  CUDA_CALL(cub::DeviceRadixSort::SortPairs(
      workspace, workspace_bytes, keys, vals, len,
      begin_bit, end_bit, cu_stream));

  workspace = device->AllocWorkspace(gpu_ctx, workspace_bytes);
  CUDA_CALL(cub::DeviceRadixSort::SortPairs(
      workspace, workspace_bytes, keys, vals, len,
      begin_bit, end_bit, cu_stream));
  device->StreamSync(gpu_ctx, stream);

  key = keys.Current();
  val = vals.Current();
  key_alter = keys.Alternate();
  val_alter = vals.Alternate();

  device->FreeWorkspace(gpu_ctx, workspace);
}

template<typename NativeKey_t, typename NativeVal_t>
void CubSortPair(
    const NativeKey_t* key_in, NativeKey_t* key_out,
    const NativeVal_t* val_in, NativeVal_t* val_out,
    const size_t num_nodes, Context gpu_ctx,
    int begin_bit = 0, int end_bit = sizeof(NativeKey_t) * 8,
    StreamHandle stream=nullptr) {
  auto cu_stream = static_cast<cudaStream_t>(stream);
  auto device = Device::Get(gpu_ctx);

  size_t workspace_bytes;
  void *workspace = nullptr;
  CUDA_CALL(cub::DeviceRadixSort::SortPairs(
      workspace, workspace_bytes, key_in, key_out, val_in, val_out, num_nodes,
      begin_bit, end_bit, cu_stream));

  workspace = device->AllocWorkspace(gpu_ctx, workspace_bytes);
  CUDA_CALL(cub::DeviceRadixSort::SortPairs(
      workspace, workspace_bytes, key_in, key_out, val_in, val_out, num_nodes,
      begin_bit, end_bit, cu_stream));
  device->StreamSync(gpu_ctx, stream);

  device->FreeWorkspace(gpu_ctx, workspace);
}

}
}
}
