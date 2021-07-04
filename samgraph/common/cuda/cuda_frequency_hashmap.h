#ifndef SAMGRAPH_CUDA_FREQUENCY_HASHMAP_H
#define SAMGRAPH_CUDA_FREQUENCY_HASHMAP_H

#include <cuda_runtime.h>

#include <cassert>
#include <cstdio>

#include "../common.h"
#include "../constant.h"

namespace samgraph {
namespace common {
namespace cuda {

class FrequencyHashmap;

class DeviceFrequencyHashmap {
 public:
  struct NodeBucket {
    IdType key;
    IdType count;  // district edge count
  };

  struct EdgeBucket {
    LongIdType key;
    IdType count;  // duplicated edge count
    IdType index;
  };

  typedef const NodeBucket *ConstNodeIterator;
  typedef const EdgeBucket *ConstEdgeIterator;

  DeviceFrequencyHashmap(const DeviceFrequencyHashmap &other) = default;
  DeviceFrequencyHashmap &operator=(const DeviceFrequencyHashmap &other) =
      default;

  inline __device__ IdType SearchNodeForPosition(const IdType id) const {
    IdType pos = NodeHash(id);

    IdType delta = 1;
    while (_node_table[pos].key != id) {
      assert(_node_table[pos].key != Constant::kEmptyKey);
      pos = NodeHash(pos + delta);
      delta += 1;
    }
    assert(pos < _ntable_size);

    return pos;
  }

  inline __device__ LongIdType SearchEdgeForPosition(const IdType src,
                                                     const IdType dst) const {
    LongIdType id = EncodeEdge(src, dst);
    LongIdType pos = EdgeHash(id);

    LongIdType delta = 1;
    while (_edge_table[pos].key != id) {
      assert(_edge_table[pos].key != Constant::kEmptyLongKey);
      pos = EdgeHash(pos + delta);
      delta += 1;
    }
    assert(pos < _etable_size);

    return pos;
  }

  inline __device__ ConstNodeIterator SearchNode(const IdType id) {
    const IdType pos = SearchNodeForPosition(id);
    return &_node_table[pos];
  }

  inline __device__ ConstEdgeIterator SearchEdge(const IdType src,
                                                 const IdType dst) {
    const LongIdType pos = SearchEdgeForPosition(src, dst);
    return &_edge_table[pos];
  }

 protected:
  const NodeBucket *_node_table;
  const EdgeBucket *_edge_table;
  const size_t _ntable_size;
  const size_t _etable_size;

  const IdType *_unique_src;
  const IdType *_unique_dst;
  const IdType *_unique_count;
  const size_t _unique_size;

  explicit DeviceFrequencyHashmap(
      const NodeBucket *node_table, const EdgeBucket *edge_table,
      const size_t ntable_size, const size_t etable_size,
      const IdType *unique_src, const IdType *unique_dst,
      const IdType *unique_count, const size_t unique_size);

  inline __device__ LongIdType EncodeEdge(const IdType src,
                                          const IdType dst) const {
    LongIdType encoding = 0;
    LongIdType id0 = (LongIdType)src;
    LongIdType id1 = (LongIdType)dst;
    encoding |= (id0 & 0xFFFFULL);
    encoding |= ((id1 & 0xFFFFULL) << 16);
    encoding |= ((id0 & 0xFFFF0000ULL) << 32);
    encoding |= ((id1 & 0xFFFF0000ULL) << 48);

    return encoding;
  }

  inline __device__ IdType NodeHash(const IdType id) const {
    return id % _ntable_size;
  };

  inline __device__ LongIdType EdgeHash(const LongIdType id) const {
    return id % _etable_size;
  };

  friend class FrequencyHashmap;
};

class FrequencyHashmap {
 public:
  static constexpr size_t kDefaultScale = 3;
  using NodeBucket = typename DeviceFrequencyHashmap::NodeBucket;
  using EdgeBucket = typename DeviceFrequencyHashmap::EdgeBucket;

  FrequencyHashmap(const size_t max_nodes, const size_t max_edges, Context ctx,
                   const size_t scale = kDefaultScale);
  ~FrequencyHashmap();

  void Reset(StreamHandle stream);
  void GetTopK(const IdType *input_src, const IdType *input_dst,
               const size_t num_input_edge, const IdType *input_nodes,
               const size_t num_input_node, const size_t K, IdType *output_src,
               IdType *output_dst, IdType *output_data, size_t *num_output,
               StreamHandle stream);
  DeviceFrequencyHashmap DeviceHandle() const;

 private:
  Context _ctx;

  NodeBucket *_node_table;
  EdgeBucket *_edge_table;
  const size_t _ntable_size;
  const size_t _etable_size;

  IdType *_node_list;
  size_t _num_node;
  const size_t _node_list_size;

  IdType *_unique_src;
  IdType *_unique_dst;
  IdType *_unique_frequency;
  size_t _num_unique;
  const size_t _unique_list_size;
};

}  // namespace cuda
}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_CUDA_FREQUENCY_HASHMAP_H