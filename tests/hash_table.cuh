#ifndef TESTS_HASH_TABEL_H
#define TESTS_HASH_TABEL_H

#include <limits>

constexpr unsigned int kEmptyKey = std::numeric_limits<unsigned int>::max();
constexpr size_t kCudaTileSize = 1024;
constexpr size_t kCudaBlockSize = 256;

class OrderedHashTable;

class DeviceOrderedHashTable {
  public:
    struct Bucket {
      unsigned int key;
      unsigned int local;
      unsigned int index;
      unsigned int version;
    };

    typedef const Bucket* ConstIterator;

    DeviceOrderedHashTable(
        const DeviceOrderedHashTable& other) = default;
    DeviceOrderedHashTable& operator=(
        const DeviceOrderedHashTable& other) = default;

    inline __device__ ConstIterator Search(const unsigned int id) const {
      const unsigned int pos = SearchForPosition(id);
      return &_table[pos];
    }

  protected:
    const Bucket * _table;
    size_t _size;
  
    explicit DeviceOrderedHashTable(
        const Bucket * table,
        size_t size);

    inline __device__ unsigned int SearchForPosition(const unsigned int id) const {
      unsigned int pos = Hash(id);

      // linearly scan for matching entry
      unsigned int delta = 1;
      while (_table[pos].key != id) {
        assert(_table[pos].key != kEmptyKey);
        pos = Hash(pos+delta);
        delta +=1;
      }
      assert(pos < _size);

      return pos;
    }

    inline __device__ unsigned int Hash(const unsigned int id) const {
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
        unsigned int device,
        cudaStream_t stream,
        const size_t scale = kDefaultScale);

    ~OrderedHashTable();

    // Disable copying 
    OrderedHashTable(
        const OrderedHashTable& other) = delete;
    OrderedHashTable& operator=(
        const OrderedHashTable& other) = delete;

    void FillWithDuplicates(
        const unsigned int * const input,
        const size_t num_input,
        unsigned int * const unique,
        size_t & num_unique,
        cudaStream_t stream);


    void FillWithUnique(
        const unsigned int * const input,
        const size_t num_input,
        cudaStream_t stream);

    DeviceOrderedHashTable DeviceHandle() const;

  private:
    Bucket * _table;
    unsigned int * _local_mapping;
    size_t _size;
    unsigned int _device;
    unsigned int _global_version;
    unsigned int _global_offset;
};

#endif // TESTS_HASH_TABEL_H