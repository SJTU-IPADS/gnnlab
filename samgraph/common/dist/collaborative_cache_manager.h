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

#pragma once

#ifndef SAMGRAPH_COLL_CACHE_MANAGER_H
#define SAMGRAPH_COLL_CACHE_MANAGER_H

#include "../common.h"
#include "../logging.h"
#include <cuda_runtime.h>

namespace samgraph {
namespace common {
namespace dist {

/**
 * @brief Coll -> Collaborative
 * a distributed cache manager that extract feature from local, remote, and cpu
 */
class CollCacheManager {
 public:
  // struct HashTableEntry{
  //   int _location_id;
  //   uint32_t _offset;
  // };
  using HashTableEntryLocation = int;
  using HashTableEntryOffset = IdType;
  struct SrcKey {
    int _location_id;
  };
  struct DstVal {
    IdType _src_offset;
    IdType _dst_offset;
  };
  CollCacheManager();
  ~CollCacheManager();

  /**
   * @brief Get the Miss Cache Index object
   *
   * @param output_miss_src_index source information for each node
   * @param output_miss_dst_index dest information for each note
   * @param nodes
   * @param num_nodes
   * @param stream
   */
  void GetMissCacheIndex(SrcKey* & output_src_index,
                         DstVal* & output_dst_index,
                         const IdType* nodes, const size_t num_nodes,
                         StreamHandle stream = nullptr);
  void SplitGroup(const SrcKey * src_index, const size_t num_node, IdType * & group_offset, StreamHandle stream = nullptr);
  void CombineOneGroup(const SrcKey * src_index, const DstVal * dst_index, const IdType* nodes, const size_t num_node, const void* src_data, void* output, StreamHandle stream = nullptr, IdType limit_block=0, bool async=false);
  template<int NUM_LINK>
  void CombineConcurrent(const SrcKey * src_index, const DstVal * dst_index, const IdType * group_offset, void* output, StreamHandle stream = nullptr);
  void CombineAllGroup(const SrcKey * src_index, const DstVal * dst_index, const IdType * group_offset, void* output, StreamHandle stream = nullptr);
  void CombineNoGroup(const IdType * nodes, const size_t num_node, void* output, Context _trainer_ctx, DataType _dtype, IdType _dim, StreamHandle stream);
  void ExtractFeat(const IdType* nodes, const size_t num_nodes,
                   void* output, StreamHandle stream = nullptr, uint64_t task_key=0xffffffffffffffff);
  inline bool IsDirectMapping() {
    if (_hash_table_location == nullptr) {
      CHECK(_num_location == 1);
      return true;
    }
    return false;
  }
  inline bool IsLegacy() {
    if (_num_location == 2) {
      CHECK(_cpu_location_id == 1);
      return true;
    }
    return false;
  }
  // typedef IdType(* OffsetGetter_t)(const SrcKey*, const DstVal*, const IdType*, const size_t);
  // typedef std::function<IdType&(SrcKey*, DstVal*, IdType*, size_t)> OffsetGetter_t;
 private:
  // template <OffsetGetter_t src_getter, OffsetGetter_t dst_getter>
  // void Combine(const SrcKey * src_index, const DstVal * dst_index, const IdType* nodes, const size_t num_node, const void* src_data, void* output, StreamHandle stream = nullptr);

  CollCacheManager(Context trainer_ctx, DataType dtype, size_t dim, int num_gpu);

  Context _trainer_ctx;

  DataType _dtype;
  size_t _dim;

  int _num_location = -1;
  int _cpu_location_id = -1;
  int _local_location_id = -1;

  size_t _cache_nbytes = 0;

  // HashTableEntry* _hash_table = nullptr;
  HashTableEntryLocation* _hash_table_location = nullptr;
  HashTableEntryOffset* _hash_table_offset = nullptr;
  std::vector<void*> _device_cache_data;

  std::vector<int> _remote_device_list;
  std::vector<int> _remote_sm_list;
  std::vector<StreamHandle> _concurrent_stream_array;

 public:
  static CollCacheManager BuildLegacy(Context trainer_ctx,
                  void* cpu_src_data, DataType dtype, size_t dim,
                  TensorPtr cache_node_tensor, size_t num_total_nodes, double cache_percentage, 
                  StreamHandle stream = nullptr);
  static CollCacheManager BuildLegacy(Context trainer_ctx,
                  void* cpu_src_data, DataType dtype, size_t dim,
                  const IdType* cache_nodes, size_t num_total_nodes, double cache_percentage,
                  StreamHandle stream = nullptr);
  static CollCacheManager BuildNoCache(Context trainer_ctx,
                  void* cpu_src_data, DataType dtype, size_t dim,
                  StreamHandle stream = nullptr);
  static CollCacheManager BuildFullCache(Context trainer_ctx,
                  void* cpu_src_data, DataType dtype, size_t dim,
                  size_t num_total_nodes,
                  StreamHandle stream = nullptr);
  static CollCacheManager BuildCollCache(
      TensorPtr node_to_block, TensorPtr block_placement, size_t num_device,
      Context trainer_ctx, 
      void* cpu_src_data, DataType dtype, size_t dim,
      int local_location_id,
      double cache_percentage, StreamHandle stream = nullptr);

  static void CheckCudaEqual(const void* a, const void* b, const size_t nbytes, StreamHandle stream = nullptr);
};

}  // namespace dist
}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_COLL_CACHE_MANAGER_H
