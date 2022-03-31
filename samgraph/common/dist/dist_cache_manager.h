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

#ifndef SAMGRAPH_DIST_CACHE_MANAGER_H
#define SAMGRAPH_DIST_CACHE_MANAGER_H

#include "../common.h"

namespace samgraph {
namespace common {
namespace dist {

class DistCacheManager {
 public:
  DistCacheManager(Context trainer_ctx,
                  const void* cpu_src_data, DataType dtype, size_t dim,
                  const IdType* nodes, size_t num_nodes,
                  double cache_percentage);
  ~DistCacheManager();

  /**
   * @brief Get the Miss Cache Index object
   *
   * @param output_miss_src_index original node id
   * @param output_miss_dst_index remapped node id(idx in `nodes`)
   * @param num_output_miss
   * @param output_cache_src_index rank of original node id
   * @param output_cache_dst_index remapped node id(idx in `nodes`)
   * @param num_output_cache
   * @param nodes
   * @param num_nodes
   * @param stream
   */
  void GetMissCacheIndex(IdType* output_miss_src_index,
                         IdType* output_miss_dst_index, size_t* num_output_miss,
                         IdType* output_cache_src_index,
                         IdType* output_cache_dst_index,
                         size_t* num_output_cache, const IdType* nodes,
                         const size_t num_nodes, StreamHandle stream = nullptr);
  void ExtractMissData(void* output_miss, const IdType* miss_src_index,
                       const size_t num_miss);
  void CombineMissData(void* output, const void* miss,
                       const IdType* miss_dst_index, const size_t num_miss,
                       StreamHandle stream);
  void CombineCacheData(void* output, const IdType* cache_src_index,
                        const IdType* cache_dst_index, const size_t num_cache,
                        StreamHandle stream);

 private:
  Context _extractor_ctx;
  Context _trainer_ctx;

  size_t _num_nodes;
  size_t _num_cached_nodes;
  double _cache_percentage;

  DataType _dtype;
  size_t _dim;

  const void* _cpu_src_data;
  size_t _cache_nbytes;
  void* _trainer_cache_data;

  IdType* _cpu_hashtable;
};

}  // namespace dist
}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_DIST_CACHE_MANAGER_H
