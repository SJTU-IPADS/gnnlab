#pragma once
#include "../common.h"
#include "../device.h"
#include "../timer.h"
#include "../logging.h"
#include <cub/cub.cuh>
namespace samgraph{
namespace common{
namespace cuda{

template<typename T>
struct SelectIdxByEqual {
  const T* d_array;
  T compare;
  CUB_RUNTIME_FUNCTION __forceinline__
  SelectIdxByEqual(const T* d_array, const T compare) : d_array(d_array), compare(compare) {}
  __host__ __device__  __forceinline__
  bool operator()(const T & idx) const {
    return d_array[idx] == compare;
  }
};

template<typename T>
void CubSelectIndexByEq(Context gpu_ctx,
    const T * d_in, const size_t num_input,
    T* d_out, size_t & num_selected_out,
    const T compare,
    StreamHandle stream = nullptr) {
  auto cu_stream = static_cast<cudaStream_t>(stream);
  auto device = Device::Get(gpu_ctx);

  cub::CountingInputIterator<T> counter(0);
  SelectIdxByEqual<T> select_op(d_in, compare);

  size_t * d_num_selected_out = Device::Get(CPU())->AllocArray<size_t>(CPU(), 1);

  size_t workspace_bytes;
  void * workspace = nullptr;
  cub::DeviceSelect::If(workspace, workspace_bytes, counter, d_out, d_num_selected_out, num_input, select_op, cu_stream);
  workspace = device->AllocWorkspace(gpu_ctx, workspace_bytes);
  cub::DeviceSelect::If(workspace, workspace_bytes, counter, d_out, d_num_selected_out, num_input, select_op, cu_stream);
  device->StreamSync(gpu_ctx, stream);
  num_selected_out = *d_num_selected_out;
  device->FreeWorkspace(gpu_ctx, workspace);
  Device::Get(CPU())->FreeWorkspace(CPU(), d_num_selected_out);
}

template<typename T>
struct EqConverter {
  T compare;
  __host__ __device__ __forceinline__
  EqConverter(const T & compare) : compare(compare) {}
  __host__ __device__ __forceinline__
  size_t operator()(const T & a) const {
    return (a == compare) ? 1 : 0;
  }
};

template<typename T>
void CubCountByEq(Context gpu_ctx,
    const T * d_in, const size_t num_input,
    size_t & count_out,
    const T compare,
    StreamHandle stream = nullptr) {
  auto cu_stream = static_cast<cudaStream_t>(stream);
  auto device = Device::Get(gpu_ctx);

  EqConverter<T> converter(compare);
  cub::TransformInputIterator<size_t, EqConverter<T>, T*> itr(const_cast<T*>(d_in), converter);

  size_t * d_count_out = Device::Get(CPU())->AllocArray<size_t>(CPU(), 1);

  size_t workspace_bytes;
  void * workspace = nullptr;
  cub::DeviceReduce::Sum(workspace, workspace_bytes, itr, d_count_out, num_input, cu_stream);
  workspace = device->AllocWorkspace(gpu_ctx, workspace_bytes);
  cub::DeviceReduce::Sum(workspace, workspace_bytes, itr, d_count_out, num_input, cu_stream);
  device->StreamSync(gpu_ctx, stream);
  count_out = *d_count_out;
  device->FreeWorkspace(gpu_ctx, workspace);
  Device::Get(CPU())->FreeWorkspace(CPU(), d_count_out);
}

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
    NativeKey_t* & key, NativeKey_t* & key_out,
    NativeVal_t* & val, NativeVal_t* & val_out,
    const size_t len, Context gpu_ctx, 
    int begin_bit = 0, int end_bit = sizeof(NativeKey_t) * 8,
    StreamHandle stream = nullptr) {
  auto cu_stream = static_cast<cudaStream_t>(stream);
  auto device = Device::Get(gpu_ctx);

  cub::DoubleBuffer<NativeKey_t> keys(key, key_out);
  cub::DoubleBuffer<NativeVal_t> vals(val, val_out);

  size_t workspace_bytes;
  void * workspace = nullptr;
  CUDA_CALL(cub::DeviceRadixSort::SortPairs(
      workspace, workspace_bytes, keys, vals, len,
      begin_bit, end_bit, cu_stream, false));

  workspace = device->AllocWorkspace(gpu_ctx, workspace_bytes);
  CUDA_CALL(cub::DeviceRadixSort::SortPairs(
      workspace, workspace_bytes, keys, vals, len,
      begin_bit, end_bit, cu_stream, false));
  device->StreamSync(gpu_ctx, stream);

  key = keys.Alternate();
  val = vals.Alternate();
  key_out = keys.Current();
  val_out = vals.Current();

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
