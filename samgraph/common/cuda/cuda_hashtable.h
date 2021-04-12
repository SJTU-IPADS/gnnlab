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
  struct Bucket {
    IdType key;
    IdType local;
    IdType index;
    IdType version;
  };

  struct Mapping {
    IdType local;
  };

  typedef const Bucket* ConstIterator;

  DeviceOrderedHashTable(const DeviceOrderedHashTable& other) = default;
  DeviceOrderedHashTable& operator=(const DeviceOrderedHashTable& other) =
      default;

  inline __device__ ConstIterator Search(const IdType id) const {
    const IdType pos = SearchForPosition(id);
    return &_table[pos];
  }

 protected:
  const Bucket* _table;
  const Mapping* _mapping;
  size_t _size;
  size_t _mapping_size;

  explicit DeviceOrderedHashTable(const Bucket* const table,
                                  const Mapping* const mapping,
                                  const size_t size, const size_t mapping_size);

  inline __device__ IdType SearchForPosition(const IdType id) const {
    IdType pos = Hash(id);

    // linearly scan for matching entry
    IdType delta = 1;
    while (_table[pos].key != id) {
      assert(_table[pos].key != Config::kEmptyKey);
      pos = Hash(pos + delta);
      delta += 1;
    }
    assert(pos < _size);

    return pos;
  }

  inline __device__ IdType Hash(const IdType id) const { return id % _size; }

  friend class OrderedHashTable;
};

class OrderedHashTable {
 public:
  static constexpr size_t kDefaultScale = 3;

  using Bucket = typename DeviceOrderedHashTable::Bucket;
  using Mapping = typename DeviceOrderedHashTable::Mapping;

  OrderedHashTable(const size_t size, int device, cudaStream_t stream,
                   const size_t scale = kDefaultScale);

  ~OrderedHashTable();

  // Disable copying
  OrderedHashTable(const OrderedHashTable& other) = delete;
  OrderedHashTable& operator=(const OrderedHashTable& other) = delete;

  void Clear(cudaStream_t stream);

  void FillWithDuplicates(const IdType* const input, const size_t num_input,
                          IdType* const unique, size_t* const num_unique,
                          cudaStream_t stream);

  void FillWithUnique(const IdType* const input, const size_t num_input,
                      cudaStream_t stream);

  size_t NumItems() const { return _offset; }

  DeviceOrderedHashTable DeviceHandle() const;

 private:
  Bucket* _table;
  Mapping* _mapping;
  size_t _size;
  size_t _mapping_size;
  int _device;

  IdType _version;
  IdType _offset;
};

}  // namespace cuda
}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_CUDA_HASHTABLE_H