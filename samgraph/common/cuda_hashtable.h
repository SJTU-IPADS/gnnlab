#ifndef SAMGPRAH_CUDA_HASHTABLE_H
#define SAMGRAPH_CUDA_HASHTABLE_H

#include <cuda_runtime.h>

#include "types.h"
#include "logging.h"

namespace samgraph {
namespace common {
namespace cuda {

class OrderedHashTable;

/*!
 * \brief A device-side handle for a GPU hashtable for mapping items to the
 * first index at which they appear in the provided data array.
 *
 * For any ID array A, one can view it as a mapping from the index `i`
 * (continuous integer range from zero) to its element `A[i]`. This hashtable
 * serves as a reverse mapping, i.e., from element `A[i]` to its index `i`.
 * Quadratic probing is used for collision resolution. See
 * DeviceOrderedHashTable's documentation for how the Mapping structure is
 * used.
 *
 * The hash table should be used in two phases, with the first being populating
 * the hash table with the OrderedHashTable object, and then generating this 
 * handle from it. This object can then be used to search the hash table,
 * to find mappings, from with CUDA code.
 *
 * If a device-side handle is created from a hash table with the following
 * entries:
 * [
 *   {key: 0, local: 0, index: 0},
 *   {key: 3, local: 1, index: 1},
 *   {key: 2, local: 2, index: 2},
 *   {key: 8, local: 3, index: 4},
 *   {key: 4, local: 4, index: 5},
 *   {key: 1, local: 5, index: 8}
 * ]
 * The array [0, 3, 2, 0, 8, 4, 3, 2, 1, 8] could have `Search()` called on
 * each id, to be mapped via:
 * ```
 * __global__ void map(int32_t * array,
 *                     size_t size,
 *                     DeviceOrderedHashTable<int32_t> table) {
 *   int idx = threadIdx.x + blockIdx.x*blockDim.x;
 *   if (idx < size) {
 *     array[idx] = table.Search(array[idx])->local;
 *   }
 * }
 * ```
 * to get the remaped array:
 * [0, 1, 2, 0, 3, 4, 1, 2, 5, 3]
 *
 */
class DeviceOrderedHashTable {
  public:
    // An entry in the hashtable.
    struct Mapping {
      // The ID of the item inserted.
      nodeid_t key;
      // The index of the item in the unique list.
      nodeid_t local;
      // The index of the item when inserted into the hashtable (e.g.,
      // the index within the array passed into FillWithDuplicates()).
      nodeid_t index;
    };

    typedef const Mapping* ConstIterator;

    DeviceOrderedHashTable(
        const DeviceOrderedHashTable& other) = default;
    DeviceOrderedHashTable& operator=(
        const DeviceOrderedHashTable& other) = default;

    /**
    * \brief Find the non-mutable mapping of a given key within the hash table.
    *
    * WARNING: The key must exist within the hashtable. Searching for a key not
    * in the hashtable is undefined behavior.
    *
    * \param id The key to search for.
    *
    * \return An iterator to the mapping.
    */
    inline __device__ ConstIterator Search(const nodeid_t id) const {
      const nodeid_t pos = SearchForPosition(id);
      return &table_[pos];
    }

  protected:
    // Must be uniform bytes for memset to work
    static constexpr nodeid_t kEmptyKey = -1;

    const Mapping * table_;
    size_t size_;

    /**
    * \brief Create a new device-side handle to the hash table.
    *
    * \param table The table stored in GPU memory.
    * \param size The size of the table.
    */
    explicit DeviceOrderedHashTable(
        const Mapping * table,
        size_t size);

    /**
    * \brief Search for an item in the hash table which is known to exist.
    *
    * WARNING: If the ID searched for does not exist within the hashtable, this
    * function will never return.
    *
    * \param id The ID of the item to search for.
    *
    * \return The the position of the item in the hashtable.
    */
    inline __device__ nodeid_t SearchForPosition(
        const nodeid_t id) const {
      nodeid_t pos = Hash(id);

      // linearly scan for matching entry
      nodeid_t delta = 1;
      while (table_[pos].key != id) {
        assert(table_[pos].key != kEmptyKey);
        pos = Hash(pos+delta);
        delta +=1;
      }
      assert(pos < size_);

      return pos;
    }

    /**
    * \brief Hash an ID to a to a position in the hash table.
    *
    * \param id The ID to hash.
    *
    * \return The hash.
    */
    inline __device__ size_t Hash(
        const nodeid_t id) const {
      return id % size_;
    }

    friend class OrderedHashTable;
};

/*!
 * \brief A host-side handle for a GPU hashtable for mapping items to the
 * first index at which they appear in the provided data array. This host-side
 * handle is responsible for allocating and free the GPU memory of the
 * hashtable.
 *
 * For any ID array A, one can view it as a mapping from the index `i`
 * (continuous integer range from zero) to its element `A[i]`. This hashtable
 * serves as a reverse mapping, i.e., from element `A[i]` to its index `i`.
 * Quadratic probing is used for collision resolution.
 *
 * The hash table should be used in two phases, the first is filling the hash
 * table via 'FillWithDuplicates()' or 'FillWithUnique()'. Then, the
 * 'DeviceHandle()' method can be called, to get a version suitable for
 * searching from device and kernel functions.
 *
 * If 'FillWithDuplicates()' was called with an array of:
 * [0, 3, 2, 0, 8, 4, 3, 2, 1, 8]
 *
 * The resulting entries in the hash-table would be:
 * [
 *   {key: 0, local: 0, index: 0},
 *   {key: 3, local: 1, index: 1},
 *   {key: 2, local: 2, index: 2},
 *   {key: 8, local: 3, index: 4},
 *   {key: 4, local: 4, index: 5},
 *   {key: 1, local: 5, index: 8}
 * ]
 *
 */
class OrderedHashTable {
  public:
    static constexpr int kDefaultScale = 3;

    using Mapping = typename DeviceOrderedHashTable::Mapping;

    /**
    * \brief Create a new ordered hash table. The amoutn of GPU memory
    * consumed by the resulting hashtable is O(`size` * 2^`scale`).
    *
    * \param size The number of items to insert into the hashtable.
    * \param scale The power of two times larger the number of buckets should
    * be than the number of items.
    */
    OrderedHashTable(
        const size_t size,
        int device,
        cudaStream_t stream,
        const int scale = kDefaultScale);

    /**
    * \brief Cleanup after the hashtable.
    */
    ~OrderedHashTable();

    // Disable copying 
    OrderedHashTable(
        const OrderedHashTable& other) = delete;
    OrderedHashTable& operator=(
        const OrderedHashTable& other) = delete;

    /**
    * \brief Fill the hashtable with the array containing possibly duplicate
    * IDs.
    *
    * \param input The array of IDs to insert.
    * \param num_input The number of IDs to insert.
    * \param unique The list of unique IDs inserted.
    * \param num_unique The number of unique IDs inserted.
    */
    void FillWithDuplicates(
        const nodeid_t * const input,
        const size_t num_input,
        nodeid_t * const unique,
        size_t * const num_unique,
        cudaStream_t stream);

    /**
    * \brief Fill the hashtable with an array of unique keys.
    *
    * \param input The array of unique IDs.
    * \param num_input The number of keys.
    */
    void FillWithUnique(
        const nodeid_t * const input,
        const size_t num_input
        cudaStream_t stream);

    /**
    * \brief Get a verison of the hashtable usable from device functions.
    * 
    * \return This hashtable.
    */
    DeviceOrderedHashTable DeviceHandle() const;

  private:
    Mapping * table_;
    size_t size_;
    int device_;
};

} // namespace cuda
} // namespace common
} // namespace samgraph

#endif // SAMGRAPH_CUDA_HASHTABLE_H