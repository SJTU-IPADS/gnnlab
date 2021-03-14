#ifndef TESTS_HASH_TABEL_H
#define TESTS_HASH_TABEL_H

#include <limits>
#include <cassert>

using IdType = unsigned int;
constexpr IdType kEmptyKey = std::numeric_limits<IdType>::max();
constexpr size_t kCudaTileSize = 1024;
constexpr size_t kCudaBlockSize = 256;

class OrderedHashTable;

class DeviceOrderedHashTable {
  public:
    struct Bucket {
      IdType key;
      IdType local;
      IdType index;
      IdType version;
    };

    typedef const Bucket* ConstIterator;

    DeviceOrderedHashTable(
        const DeviceOrderedHashTable& other) = default;
    DeviceOrderedHashTable& operator=(
        const DeviceOrderedHashTable& other) = default;

    inline __device__ ConstIterator Search(const IdType id) const {
      const IdType pos = SearchForPosition(id);
      return &_table[pos];
    }

  protected:
    const Bucket * _table;
    size_t _size;
  
    explicit DeviceOrderedHashTable(
        const Bucket * const table,
        const size_t size);

    inline __device__ IdType SearchForPosition(const IdType id) const {
      IdType pos = Hash(id);

      // linearly scan for matching entry
      IdType delta = 1;
      while (_table[pos].key != id) {
        assert(_table[pos].key != kEmptyKey);
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

    using Bucket = typename DeviceOrderedHashTable::Bucket;

    OrderedHashTable(
        const size_t size,
        IdType device,
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
        size_t & num_unique,
        cudaStream_t stream);


    void FillWithUnique(
        const IdType * const input,
        const size_t num_input,
        cudaStream_t stream);

    DeviceOrderedHashTable DeviceHandle() const;

  private:
    Bucket * _table;
    IdType * _local_mapping;
    size_t _size;
    IdType _device;
    IdType _global_version;
    IdType _global_offset;
};

#endif // TESTS_HASH_TABEL_H