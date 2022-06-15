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

#ifndef SAMGRAPH_CUDA_UTILS_H
#define SAMGRAPH_CUDA_UTILS_H

#include "../common.h"
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

namespace samgraph {
namespace common {
namespace cuda {

/**
 * This structure is used with cub's block-level prefixscan in order to
 * keep a running sum as items are iteratively processed.
 */
template <typename T>
struct BlockPrefixCallbackOp {
  T _running_total;

  __device__ BlockPrefixCallbackOp(const T running_total)
      : _running_total(running_total) {}

  __device__ T operator()(const T block_aggregate) {
    const T old_prefix = _running_total;
    _running_total += block_aggregate;
    return old_prefix;
  }
};

template<typename T>
void ArrangeArray(T* array, size_t array_len, T begin = 0, T step = 1, StreamHandle = nullptr);

// always with replacement
template<typename T>
void UniformFill(T* array, size_t array_len, StreamHandle = nullptr);

void GPUGetRowFromEid(const IdType * indptr, size_t n_row, const IdType *eid_list, size_t num_eid, IdType * output_row, StreamHandle stream);

class ArrayGenerator {
 public:
  template<typename T>
  void byArrange(T* array, size_t array_len, T begin = 0, T step = 1, StreamHandle stream = nullptr) {
    ArrangeArray(array, array_len, begin, step, stream);
  }
  template<typename T>
  void byRepeat(T* array, const T* src_array, size_t src_len, const size_t repeats, StreamHandle stream = nullptr);


  template<typename T>
  void byUniform(T* array, size_t array_len, T min, T max, curandState *random_states, size_t num_random_states, StreamHandle = nullptr);

  template<typename T>
  void byWeightAlias(T* array, size_t array_len, StreamHandle = nullptr) = delete;
  void InitWeightAlias(TensorPtr weight) = delete;

  template<typename T>
  void byWeightPrefix(T* array, size_t array_len, StreamHandle = nullptr) = delete;
  void InitWeightPrefix(TensorPtr weight) = delete;
 private:
  TensorPtr prob, alias;
};

}  // namespace cuda
}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_CUDA_UTILS_H