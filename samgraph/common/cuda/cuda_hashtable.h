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
  struct BucketO2N {
    IdType key;
    IdType local;
    IdType index;
    IdType version;
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

  explicit DeviceOrderedHashTable(const BucketO2N *const o2n_table,
                                  const BucketN2O *const n2o_table,
                                  const size_t o2n_size, const size_t n2o_size);

  inline __device__ IdType SearchForPositionO2N(const IdType id) const {
    IdType pos = HashO2N(id);

    // linearly scan for matching entry
    IdType delta = 1;
    while (_o2n_table[pos].key != id) {
      assert(_o2n_table[pos].key != Constant::kEmptyKey);
      pos = HashO2N(pos + delta);
      delta += 1;
    }
    assert(pos < _o2n_size);

    return pos;
  }

  inline __device__ IdType HashO2N(const IdType id) const {
    return id % _o2n_size;
  }

  friend class OrderedHashTable;
};

class OrderedHashTable {
 public:
  static constexpr size_t kDefaultScale = 3;

  using BucketO2N = typename DeviceOrderedHashTable::BucketO2N;
  using BucketN2O = typename DeviceOrderedHashTable::BucketN2O;

  OrderedHashTable(const size_t size, Context ctx,
                   const size_t scale = kDefaultScale);

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
  void CopyUnique(IdType * const unique, StreamHandle stream);
  void RefUnique(const IdType * &unique, IdType * const num_unique);
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