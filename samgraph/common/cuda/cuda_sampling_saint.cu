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

#include <curand.h>
#include <curand_kernel.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cub/cub.cuh>

#include <cusparse.h>

#include "../engine.h"
#include "../common.h"
#include "../constant.h"
#include "../device.h"
#include "../logging.h"
#include "../profiler.h"
#include "../timer.h"
#include "cuda_function.h"
#include "cuda_utils.h"
#include "cuda_hashtable.h"

namespace samgraph {
namespace common {
namespace cuda {

namespace {

#define SAM_1D_GRID_INIT(num_input) \
  const size_t num_tiles = RoundUpDiv((size_t)num_input, Constant::kCudaTileSize); \
  const dim3 grid(num_tiles);                                              \
  const dim3 block(Constant::kCudaBlockSize);                              \

#define SAM_1D_GRID_FOR(loop_iter, num_input) \
  assert(BLOCK_SIZE == blockDim.x);                       \
  const size_t block_start = TILE_SIZE * blockIdx.x;      \
  const size_t block_end = min(TILE_SIZE * (blockIdx.x + 1), num_input);  \
  for (size_t loop_iter = threadIdx.x + block_start; loop_iter < block_end; loop_iter += BLOCK_SIZE) \

__device__ IdType _UpperBound(const IdType *A, int64_t n, IdType x) {
  IdType l = 0, r = n, m = 0;
  while (l < r) {
    m = l + (r-l)/2;
    if (x >= A[m]) {
      l = m+1;
    } else {
      r = m;
    }
  }
  return l;
}

template <size_t BLOCK_SIZE=Constant::kCudaBlockSize, size_t TILE_SIZE=Constant::kCudaTileSize>
__global__ void _CSRGetRowNNZKernel(
    const IdType* indptr,
    const IdType* input_nodes,
    size_t num_input,
    IdType* out) {

  SAM_1D_GRID_FOR(index, num_input) {
    out[index] = indptr[input_nodes[index] + 1] - indptr[input_nodes[index]];
  }
}

void GetNNZ(const IdType * indptr, const IdType * input_nodes, const size_t num_input, IdType* output, StreamHandle stream ) {
  SAM_1D_GRID_INIT(num_input);
  auto cu_stream = (cudaStream_t)(stream);
  _CSRGetRowNNZKernel<><<<grid, block, 0, cu_stream>>>(indptr, input_nodes, num_input, output);
}

template <typename DType, size_t BLOCK_SIZE=Constant::kCudaBlockSize, size_t TILE_SIZE=Constant::kCudaTileSize>
__global__ void _SegmentCopyKernel(
    const IdType* indptr, const DType* data,
    const IdType* selected_row, int64_t length, int64_t n_row,
    const IdType* out_indptr, DType* out_data) {
  SAM_1D_GRID_FOR(index, length) {
    // do a binary search to identify corresponding selected_row in output indptr
    IdType rpos = _UpperBound(out_indptr, n_row, index) - 1;
    // offset in the selected_row
    IdType rofs = index - out_indptr[rpos];
    // original vertex
    const IdType u = selected_row[rpos];
    out_data[index] = data[indptr[u]+rofs];
  }
}

// /** 
//  * Must ensure that degree_list & indptr_output is of size "num_vertex + 1"
//  */
// void DegToIndptr(IdType * degree_list, IdType * indptr_output, size_t num_vertex,
//     Context ctx, StreamHandle stream) {
//   auto cu_stream = (cudaStream_t)(stream);
//   auto device = Device::Get(ctx);

//   size_t workspace_bytes;
//   cub::DeviceScan::ExclusiveSum(nullptr, workspace_bytes, indptr_output, indptr_output, num_vertex + 1, cu_stream);
//   void * workspace = device->AllocWorkspace(ctx, workspace_bytes);
//   cub::DeviceScan::ExclusiveSum(workspace, workspace_bytes, indptr_output, indptr_output, num_vertex + 1, cu_stream);

// }

void CSRSliceRows(const IdType* indptr, const IdType* indices, const IdType* input_nodes, const IdType num_input,
    TensorPtr & indptr_output_tensor, TensorPtr & indices_output_tensor,
    Context ctx, StreamHandle stream) {
  indptr_output_tensor = Tensor::Empty(DataType::kI32, {num_input + 1}, ctx, "");
  IdType* indptr_output = indptr_output_tensor->Ptr<IdType>();
  GetNNZ(indptr, input_nodes, num_input, indptr_output, stream);
  auto cu_stream = (cudaStream_t)(stream);
  auto device = Device::Get(ctx);

  size_t workspace_bytes;
  cub::DeviceScan::ExclusiveSum(nullptr, workspace_bytes, indptr_output, indptr_output, num_input + 1, cu_stream);
  void * workspace = device->AllocWorkspace(ctx, workspace_bytes);
  cub::DeviceScan::ExclusiveSum(workspace, workspace_bytes, indptr_output, indptr_output, num_input + 1, cu_stream);

  IdType nnz = 0;
  device->CopyDataFromTo(indptr_output + num_input, 0, &nnz, 0, sizeof(IdType), ctx, CPU(), stream);
  device->StreamSync(ctx, stream);
  device->FreeWorkspace(ctx, workspace);

  // Copy indices.
  indices_output_tensor = Tensor::Empty(DataType::kI32, {nnz + 1}, ctx, "");
  SAM_1D_GRID_INIT(nnz);
  _SegmentCopyKernel<><<<grid, block, 0, cu_stream>>>(indptr, indices, input_nodes, nnz, num_input, indptr_output, indices_output_tensor->Ptr<IdType>());
}


template <size_t BLOCK_SIZE=Constant::kCudaBlockSize, size_t TILE_SIZE=Constant::kCudaTileSize>
__global__ void _SegmentSelectRemapColKernel(
    const IdType* indptr, const IdType* indices, int64_t num_rows, int64_t num_nnz,
    DeviceOrderedHashTable hashtable,
    IdType* remapped_col, IdType* count) {
  SAM_1D_GRID_FOR(tx, num_nnz) {
    IdType rpos = _UpperBound(indptr, num_rows, tx) - 1;
    // IdType cur_c = indices[tx];
    // IdType i = dgl::cuda::_BinarySearch(col, col_len, cur_c);
    IdType remapped_col_id = hashtable.SearchO2NToLocal(indices[tx]);
    remapped_col[tx] = remapped_col_id;
    if (remapped_col_id != Constant::kEmptyKey) {
      atomicAdd(count+rpos, IdType(1));
    }
  }
}

// template <size_t BLOCK_SIZE=Constant::kCudaBlockSize, size_t TILE_SIZE=Constant::kCudaTileSize>
// __global__ void _SegmentMaskColKernel(
//     const IdType* indptr, const IdType* indices, int64_t num_rows, int64_t num_nnz,
//     DeviceOrderedHashTable * hashtable,
//     IdType* mask, IdType* count) {
//   SAM_1D_GRID_FOR(tx, num_nnz) {
//     IdType rpos = _UpperBound(indptr, num_rows, tx) - 1;
//     // IdType cur_c = indices[tx];
//     // IdType i = dgl::cuda::_BinarySearch(col, col_len, cur_c);
//     if (hashtable->SearchO2NIfExist(indices[tx])) {
//       mask[tx] = 1;
//       atomicAdd(count+rpos, IdType(1));
//     }
//   }
// }

struct IsNotEmptyKey {
  __device__ bool operator()(const IdType & in) {
    return in != Constant::kEmptyKey;
  }
};


inline __device__ size_t to_pos(size_t node_idx, size_t num_walk, size_t walk_len, size_t step_idx, size_t walk_idx) {
  return node_idx * num_walk * walk_len + step_idx * num_walk + walk_idx;
}

__global__ void sample_random_walk(
    const IdType *indptr, const IdType *indices, const IdType *input,
    const size_t num_input, const size_t random_walk_length,
    const double restart_prob, const size_t num_random_walk,
    IdType *frontier, curandState *random_states, size_t num_random_states) {
  size_t thread_id = blockDim.x * blockDim.y * blockIdx.x +
                     blockDim.y * threadIdx.x + threadIdx.y;
  assert(thread_id < num_random_states);
  curandState local_state = random_states[thread_id];

  size_t node_idx = blockIdx.x * blockDim.y + threadIdx.y;
  const size_t stride = blockDim.y * gridDim.x;
  /** SXN: this loop is also useless*/
  while (node_idx < num_input) {
    IdType start_node = input[node_idx];
    size_t random_walk_idx = threadIdx.x;
    while (random_walk_idx < num_random_walk) {
      IdType node = start_node;
      size_t pos = to_pos(node_idx, num_random_walk, random_walk_length, 0, random_walk_idx);
      // frontier[pos] = node;
      for (size_t step_idx = 0; step_idx < random_walk_length; step_idx++) {
        /*
         *  Get the position on the output position of random walk
         *  suppose that num_random_walk = 2, num_input = 2
         *
         *  layout:
         *     ----[root step of walk 0 of node 0]
         *     ----[root step of walk 1 of node 0]
         *     [first step of walk 0 of node 0]
         *     [first step of walk 1 of node 0]
         *     [second step of walk 0 of node 0]
         *     [second step of walk 1 of node 0]
         *     ----[root step of walk 0 of node 1]
         *     ----[root step of walk 1 of node 1]
         *     [first step of walk 0 of node 1]
         *     [first step of walk 1 of node 1]
         *     [second step of walk 0 of node 1]
         *     [second step of walk 1 of node 1]
         *     ......
         */
        // size_t pos = node_idx * num_random_walk * random_walk_length +
        //              step_idx * num_random_walk + random_walk_idx;
        size_t pos = to_pos(node_idx, num_random_walk, random_walk_length, step_idx, random_walk_idx);

        if (node != Constant::kEmptyKey) {
          const IdType off = indptr[node];
          const IdType len = indptr[node + 1] - indptr[node];
          if (len == 0) {
            node = Constant::kEmptyKey;
            frontier[pos] = start_node; // avoid compact!
          } else {
            size_t k = curand(&local_state) % len;
            node = indices[off + k];
            frontier[pos] = node;
          }
        } else {
          frontier[pos] = start_node;
        }
        // terminate
        if (restart_prob != 0) {
          if (curand_uniform_double(&local_state) < restart_prob) {
            node = Constant::kEmptyKey;
          }
        }
      }

      random_walk_idx += blockDim.x;
    }

    node_idx += stride;
  }
  // restore the state
  random_states[thread_id] = local_state;
}

}  // namespace

void GPUSampleSaintWalk(const IdType *indptr, const IdType *indices,
                         const IdType *input, const size_t num_input,
                         const size_t random_walk_length,
                         const double random_walk_restart_prob,
                         const size_t num_random_walk,
                         IdType *out_dst,
                        //  IdType *out_data,
                         size_t &num_samples,
                         Context ctx, StreamHandle stream,
                         GPURandomStates *random_states, uint64_t task_key) {
  auto sampler_device = Device::Get(ctx);
  auto cu_stream = static_cast<cudaStream_t>(stream);
  num_samples = num_input * num_random_walk * (random_walk_length);

  // 1. random walk sampling
  Timer t0;
  // IdType *tmp_dst = static_cast<IdType *>(sampler_device->AllocWorkspace(ctx, sizeof(IdType) * num_samples));

  dim3 block(Constant::kCudaBlockSize, 1);
  while (static_cast<size_t>(block.x) >= 2 * num_random_walk) {
    block.x /= 2;
    block.y *= 2;
  }
  const dim3 grid(RoundUpDiv(num_input, static_cast<size_t>(block.y)));

  sample_random_walk<<<grid, block, 0, cu_stream>>>(
      indptr, indices, input, num_input, random_walk_length,
      random_walk_restart_prob, num_random_walk, out_dst,
      random_states->GetStates(), random_states->NumStates());

  sampler_device->StreamSync(ctx, stream);
  double random_walk_sampling_time = t0.Passed();
  // 2. Only a simple dedup is required?

  // // 2. TopK
  // Timer t1;
  // frequency_hashmap->GetTopK(tmp_src, tmp_dst, num_samples, input, num_input, K,
  //                            out_src, out_dst, out_data, num_out, stream,
  //                            task_key);

  // sampler_device->FreeWorkspace(ctx, tmp_dst);
  // double topk_time = t1.Passed();

  Profiler::Get().LogStepAdd(task_key, kLogL3RandomWalkSampleCooTime,
                             random_walk_sampling_time);
  // Profiler::Get().LogStepAdd(task_key, kLogL3RandomWalkTopKTime, topk_time);
}


void CSRSliceMatrix(const IdType* indptr, const IdType* indices, const IdType* input_nodes, const IdType num_input,
    DeviceOrderedHashTable hashtable,
    TensorPtr & matrix_slice_indptr, TensorPtr & matrix_slice_indices, TensorPtr & matrix_slice_coo_row, 
    Context ctx, StreamHandle stream) {
  auto cu_stream = (cudaStream_t)(stream);
  if (num_input == 0) {
    CHECK(false);
  }

  // First slice rows
  TensorPtr row_slice_indptr, row_slice_indices;
  CSRSliceRows(indptr, indices, input_nodes, num_input, row_slice_indptr, row_slice_indices, ctx, stream);

  if (row_slice_indptr->Shape()[0] == 0) {
    CHECK(false);
  }

  // Generate a 0-1 mask for matched (row, col) positions.
  TensorPtr remapped_col_mark = Tensor::Empty(kI32, {row_slice_indices->Shape()[0]}, ctx, "");
  // A count for how many masked values per row.
  TensorPtr count = Tensor::Empty(kI32, {num_input + 1}, ctx, "");
  CUDA_CALL(cudaMemsetAsync(count->Ptr<IdType>(), 0, count->NumBytes(), cu_stream));

  const int64_t nnz_row_slice = row_slice_indices->Shape()[0];

  auto device = Device::Get(ctx);
  // Execute SegmentMaskColKernel
  SAM_1D_GRID_INIT(nnz_row_slice);
  _SegmentSelectRemapColKernel<><<<grid, block, 0, cu_stream>>>(
      row_slice_indptr->Ptr<IdType>(), row_slice_indices->Ptr<IdType>(), num_input, nnz_row_slice,
      hashtable,
      remapped_col_mark->Ptr<IdType>(), count->Ptr<IdType>());

  IdType nnz_matrix_slice = 0;
  {
    size_t workspace_bytes;
    void * workspace = nullptr;
    cub::DeviceScan::ExclusiveSum(workspace, workspace_bytes, count->Ptr<IdType>(), count->Ptr<IdType>(), num_input + 1, cu_stream);
    workspace = device->AllocWorkspace(ctx, workspace_bytes);
    cub::DeviceScan::ExclusiveSum(workspace, workspace_bytes, count->Ptr<IdType>(), count->Ptr<IdType>(), num_input + 1, cu_stream);

    device->CopyDataFromTo(count->Ptr<IdType>() + num_input, 0, &nnz_matrix_slice, 0, sizeof(IdType), ctx, CPU(), stream);
    device->StreamSync(ctx, stream);
    device->FreeWorkspace(ctx, workspace);
  }
  matrix_slice_indptr = count;
  matrix_slice_indices = Tensor::Empty(kI32, {nnz_matrix_slice}, ctx, "");
  matrix_slice_coo_row = Tensor::Empty(kI32, {nnz_matrix_slice}, ctx, "");

  if (nnz_matrix_slice == 0) {
    CHECK(false);
  }

  {
    size_t workspace_bytes = 0;
    uint8_t * workspace = nullptr;
    IdType* num_in_gpu = nullptr;
    cub::DeviceSelect::If     (workspace, workspace_bytes, remapped_col_mark->Ptr<IdType>(), matrix_slice_indices->Ptr<IdType>(),                      num_in_gpu, remapped_col_mark->Shape()[0], IsNotEmptyKey(), cu_stream);
    // cub::DeviceSelect::Flagged(workspace, workspace_bytes, row_slice_indices->Ptr<IdType>(), mask->Ptr<IdType>(), matrix_slice_indices->Ptr<IdType>(), num_in_gpu, mask->Shape()[0], cu_stream);
    workspace_bytes = ((workspace_bytes + 3) & (~0x03));
    workspace = (uint8_t*)device->AllocWorkspace(ctx, workspace_bytes + sizeof(IdType));
    num_in_gpu = (IdType*)(workspace + workspace_bytes);
    cub::DeviceSelect::If     (workspace, workspace_bytes, remapped_col_mark->Ptr<IdType>(), matrix_slice_indices->Ptr<IdType>(),                      num_in_gpu, remapped_col_mark->Shape()[0], IsNotEmptyKey(), cu_stream);
    // cub::DeviceSelect::Flagged(workspace, workspace_bytes, row_slice_indices->Ptr<IdType>(), mask->Ptr<IdType>(), matrix_slice_indices->Ptr<IdType>(), num_in_gpu, mask->Shape()[0], cu_stream);

    IdType nnz;
    device->CopyDataFromTo(num_in_gpu, 0, &nnz, 0, sizeof(IdType), ctx, CPU(), stream);
    device->StreamSync(ctx, stream);
    device->FreeWorkspace(ctx, workspace);
    CHECK_EQ(nnz, nnz_matrix_slice);
  }

  {
    cusparseXcsr2coo(Engine::Get()->GetSparseHandle(), matrix_slice_indptr->Ptr<int>(), nnz_matrix_slice, num_input, matrix_slice_coo_row->Ptr<int>(), cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO);
  }

  // // Relabel column
  // IdArray col_hash = NewIdArray(csr.num_cols, ctx, nbits);
  // Scatter_(cols, Range(0, cols->shape[0], nbits, ctx), col_hash);
  // ret_col = IndexSelect(col_hash, ret_col);
}

}  // namespace cuda
}  // namespace common
}  // namespace samgraph
