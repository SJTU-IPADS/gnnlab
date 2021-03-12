#ifndef SAMGRAPH_CUDA_HASHTABLE_H
#define SAMGRAPH_CUDA_HASHTABLE_H

#include <cassert>

#include <cuda_runtime.h>

#include "types.h"
#include "logging.h"
#include "config.h"

namespace samgraph {
namespace common {
namespace cuda {

class OrderedHashTable;

class DeviceOrderedHashTable {
  public:
    struct Mapping {
      IdType key;
      IdType local;
      IdType index;
    };

    typedef const Mapping* ConstIterator;

    DeviceOrderedHashTable(
        const DeviceOrderedHashTable& other) = default;
    DeviceOrderedHashTable& operator=(
        const DeviceOrderedHashTable& other) = default;

    inline __device__ ConstIterator Search(const IdType id) const {
      const IdType pos = SearchForPosition(id);
      return &_table[pos];
    }

  protected:
    const Mapping * _table;
    size_t _size;

    explicit DeviceOrderedHashTable(
        const Mapping * table,
        size_t size);

    inline __device__ IdType SearchForPosition(const IdType id) const {
      IdType pos = Hash(id);

      // linearly scan for matching entry
      IdType delta = 1;
      while (_table[pos].key != id) {
        assert(_table[pos].key != Config::kEmptyKey);
        pos = Hash(pos+delta);
        delta +=1;
      }
      assert(pos < _size);

      return pos;
    }

    inline __device__ IdType Hash(const IdType id) const {
      return id % _size;
    }

    friend class OrderedHashTable;
};

class OrderedHashTable {
  public:
    static constexpr size_t kDefaultScale = 3;

    using Mapping = typename DeviceOrderedHashTable::Mapping;

    OrderedHashTable(
        const size_t size,
        int device,
        cudaStream_t stream,
        const size_t scale = kDefaultScale);

    ~OrderedHashTable();

    // Disable copying 
    OrderedHashTable(
        const OrderedHashTable& other) = delete;
    OrderedHashTable& operator=(
        const OrderedHashTable& other) = delete;

    void FillWithDuplicates(
        const IdType * const input,
        const size_t num_input,
        IdType * const unique,
        size_t * const num_unique,
        cudaStream_t stream);

    void FillWithUnique(
        const IdType * const input,
        const size_t num_input,
        cudaStream_t stream);

    DeviceOrderedHashTable DeviceHandle() const;

  private:
    Mapping * _table;
    size_t _size;
    int _device;
};

} // namespace cuda
} // namespace common
} // namespace samgraph

#endif // SAMGRAPH_CUDA_HASHTABLE_H