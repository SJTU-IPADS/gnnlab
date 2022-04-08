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
  using NodeBucket = IdType;

  struct EdgeBucket {
    IdType key;
    IdType count;  // duplicated edge count
    IdType index;
  };

  typedef const NodeBucket *ConstNodeIterator;
  typedef const EdgeBucket *ConstEdgeIterator;

  DeviceFrequencyHashmap(const DeviceFrequencyHashmap &other) = default;
  DeviceFrequencyHashmap &operator=(const DeviceFrequencyHashmap &other) =
      default;

  /** SXN: a fall back path? avoid potential infinite loop */
  inline __device__ IdType SearchEdgeForPosition(const IdType node_idx,
                                                 const IdType dst) const {
    IdType start_off = node_idx * _per_node_etable_size;
    IdType pos = EdgeHash(dst);

    IdType delta = 1;
    while (_edge_table[start_off + pos].key != dst) {
      pos = EdgeHash(pos + delta);
      delta += 1;
    }
    assert(start_off + pos < _etable_size);

    return start_off + pos;
  }

  inline __device__ ConstNodeIterator SearchNode(const IdType id) {
    return &_node_table[id];
  }

  inline __device__ ConstEdgeIterator SearchEdge(const IdType node_idx,
                                                 const IdType dst) {
    const IdType pos = SearchEdgeForPosition(node_idx, dst);
    return &_edge_table[pos];
  }
  inline __device__ IdType PosToNodeIdx(const IdType pos) const {
    return pos / _per_node_etable_size;
  }

 protected:
  const NodeBucket *_node_table;
  const EdgeBucket *_edge_table;
  const size_t _ntable_size;
  const size_t _etable_size;
  const size_t _per_node_etable_size;

  explicit DeviceFrequencyHashmap(
      const NodeBucket *node_table, const EdgeBucket *edge_table,
      const size_t ntable_size, const size_t etable_size,
      const size_t per_node_etable_size);

  inline __device__ IdType EdgeHash(const IdType id) const {
    return id % _per_node_etable_size;
  };

  friend class FrequencyHashmap;
};

class FrequencyHashmap {
 public:
  static constexpr size_t kDefaultNodeTableScale = 3;
  static constexpr size_t kDefaultEdgeTableScale = 3;
  using NodeBucket = typename DeviceFrequencyHashmap::NodeBucket;
  using EdgeBucket = typename DeviceFrequencyHashmap::EdgeBucket;

  FrequencyHashmap(const size_t max_nodes, const size_t edges_per_node,
                   Context ctx,
                   const size_t node_table_scale = kDefaultNodeTableScale,
                   const size_t edge_table_scale = kDefaultEdgeTableScale);
  ~FrequencyHashmap();

  void GetTopK(IdType *input_src, IdType *input_dst,
               const size_t num_input_edge, const IdType *input_nodes,
               const size_t num_input_node, const size_t K, IdType *output_src,
               IdType *output_dst, IdType *output_data, size_t *num_output,
               StreamHandle stream, uint64_t task_key);
  DeviceFrequencyHashmap DeviceHandle() const;

 private:
  Context _ctx;

  const size_t _max_nodes;
  const size_t _edges_per_node;

  NodeBucket *_node_table;
  EdgeBucket *_edge_table;
  const size_t _ntable_size;
  const size_t _etable_size;
  const size_t _per_node_etable_size;

  size_t _num_node;
  const size_t _node_list_size;


  IdType *_unique_node_idx;
  IdType *_unique_dst;
  Id64Type *_unique_combination_key;  // 63:32 (num_node - node_idx)  31:0 frequency
  size_t _num_unique;
  const size_t _unique_list_size;
};

}  // namespace cuda
}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_CUDA_FREQUENCY_HASHMAP_H
