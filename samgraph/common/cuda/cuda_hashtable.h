#ifndef SAMGRAPH_CUDA_HASHTABLE_H
#define SAMGRAPH_CUDA_HASHTABLE_H

#include <cuda_runtime.h>

#include <cassert>

#include "../common.h"
#include "../config.h"
#include "../logging.h"

namespace samgraph {
namespace common {
namespace cuda {

class OrderedHashTable;

class DeviceOrderedHashTable {
 public:
  struct Bucket0 {
    IdType key;
    IdType local;
    IdType index;
    IdType version;
  };

  struct Bucket1 {
    IdType global;
  };

  typedef const Bucket0 *ConstIterator;

  DeviceOrderedHashTable(const DeviceOrderedHashTable &other) = default;
  DeviceOrderedHashTable &operator=(const DeviceOrderedHashTable &other) =
      default;

  inline __device__ ConstIterator Search0(const IdType id) const {
    const IdType pos = SearchForPosition0(id);
    return &_o2n_table[pos];
  }

 protected:
  const Bucket0 *_o2n_table;
  const Bucket1 *_n2o_table;
  size_t _o2n_size;
  size_t _n2o_size;

  explicit DeviceOrderedHashTable(const Bucket0 *const o2n_table,
                                  const Bucket1 *const n2o_table,
                                  const size_t o2n_size, const size_t n2o_size);

  inline __device__ IdType SearchForPosition0(const IdType id) const {
    IdType pos = Hash0(id);

    // linearly scan for matching entry
    IdType delta = 1;
    while (_o2n_table[pos].key != id) {
      assert(_o2n_table[pos].key != Config::kEmptyKey);
      pos = Hash0(pos + delta);
      delta += 1;
    }
    assert(pos < _o2n_size);

    return pos;
  }

  inline __device__ IdType Hash0(const IdType id) const {
    return id % _o2n_size;
  }

  friend class OrderedHashTable;
};

class OrderedHashTable {
 public:
  static constexpr size_t kDefaultScale = 3;

  using Bucket0 = typename DeviceOrderedHashTable::Bucket0;
  using Bucket1 = typename DeviceOrderedHashTable::Bucket1;

  OrderedHashTable(const size_t size, Context ctx, StreamHandle stream,
                   const size_t scale = kDefaultScale);

  ~OrderedHashTable();

  // Disable copying
  OrderedHashTable(const OrderedHashTable &other) = delete;
  OrderedHashTable &operator=(const OrderedHashTable &other) = delete;

  void Reset(StreamHandle stream);

  void FillWithDuplicates(const IdType *const input, const size_t num_input,
                          IdType *const unique, size_t *const num_unique,
                          StreamHandle stream);

  void FillWithUnique(const IdType *const input, const size_t num_input,
                      StreamHandle stream);

  size_t NumItems() const { return _num_items; }

  DeviceOrderedHashTable DeviceHandle() const;

 private:
  Context _ctx;

  Bucket0 *_o2n_table;
  Bucket1 *_n2o_table;
  size_t _o2n_size;
  size_t _n2o_size;

  IdType _version;
  IdType _num_items;
};

}  // namespace cuda
}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_CUDA_HASHTABLE_H