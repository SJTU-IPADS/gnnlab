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

#ifndef SAMGRAPH_CUDA_HASHTABLE_H
#define SAMGRAPH_CUDA_HASHTABLE_H

#include <cuda_runtime.h>

#include <cassert>

#include "../common.h"
#include "../constant.h"
// #include "../logging.h"

namespace samgraph {
namespace common {
namespace cuda {

class OrderedHashTable;

class DeviceOrderedHashTable {
 public:
  struct alignas(unsigned long long) BucketO2N {
    // don't change the position of version and key
    //   which used for efficient insert operation
    IdType version;
    IdType key;
    IdType local;
    IdType index;
  };

  struct BucketN2O {
    IdType global;
  };

  typedef const BucketO2N *ConstIterator;

  DeviceOrderedHashTable(const DeviceOrderedHashTable &other) = default;
  DeviceOrderedHashTable &operator=(const DeviceOrderedHashTable &other) =
      default;

  inline __device__ ConstIterator SearchO2N(const IdType id) const {
    const IdType pos = SearchForPositionO2N(id);
    return &_o2n_table[pos];
  }

 protected:
  const BucketO2N *_o2n_table;
  const BucketN2O *_n2o_table;
  const size_t _o2n_size;
  const size_t _n2o_size;
  IdType _version;

  explicit DeviceOrderedHashTable(const BucketO2N *const o2n_table,
                                  const BucketN2O *const n2o_table,
                                  const size_t o2n_size, const size_t n2o_size,
                                  const IdType version);

  inline __device__ IdType SearchForPositionO2N(const IdType id) const {
#ifndef SXN_NAIVE_HASHMAP
    IdType pos = HashO2N(id);

    // linearly scan for matching entry
    IdType delta = 1;
    while (_o2n_table[pos].key != id) {
      assert(_o2n_table[pos].version == this->_version);
      pos = HashO2N(pos + delta);
      delta += 1;
    }
    assert(pos < _o2n_size);

    return pos;
#else
    return id;
#endif
  }

  inline __device__ IdType HashO2N(const IdType id) const {
#ifndef SXN_NAIVE_HASHMAP
    return id % _o2n_size;
#else
    return id;
#endif
  }

  friend class OrderedHashTable;
};

class OrderedHashTable {
 public:
  static constexpr size_t kDefaultScale = 2;

  using BucketO2N = typename DeviceOrderedHashTable::BucketO2N;
  using BucketN2O = typename DeviceOrderedHashTable::BucketN2O;

  OrderedHashTable(const size_t size, Context ctx,
                   StreamHandle stream, const size_t scale = kDefaultScale);

  ~OrderedHashTable();

  // Disable copying
  OrderedHashTable(const OrderedHashTable &other) = delete;
  OrderedHashTable &operator=(const OrderedHashTable &other) = delete;

  void Reset(StreamHandle stream);

  void FillWithDuplicates(const IdType *const input, const size_t num_input,
                          IdType *const unique, IdType *const num_unique,
                          StreamHandle stream);

  void FillWithDupRevised(const IdType *const input, const size_t num_input,
                          // IdType *const unique, IdType *const num_unique,
                          StreamHandle stream);
  void FillWithDupMutable(IdType *const input, const size_t num_input,
                          StreamHandle stream);
  void CopyUnique(IdType * const unique, StreamHandle stream);
  void RefUnique(const IdType * &unique, IdType * const num_unique);
  /** add all neighbours of nodes in hashtable to hashtable */
  void FillNeighbours(const IdType *const indptr, const IdType *const indices,
                      StreamHandle stream);

  void FillWithUnique(const IdType *const input, const size_t num_input,
                      StreamHandle stream);

  size_t NumItems() const { return _num_items; }

  DeviceOrderedHashTable DeviceHandle() const;

 private:
  Context _ctx;

  BucketO2N *_o2n_table;
  BucketN2O *_n2o_table;
  size_t _o2n_size;
  size_t _n2o_size;

  IdType _version;
  IdType _num_items;
};

}  // namespace cuda
}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_CUDA_HASHTABLE_H
