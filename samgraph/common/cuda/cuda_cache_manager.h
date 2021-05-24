#ifndef SAMGRAPH_CUDA_CACHE_MANAGER_H
#define SAMGRAPH_CUDA_CACHE_MANAGER_H

#include "../common.h"

namespace samgraph {
namespace common {
namespace cuda {

class GPUCacheManager {
 public:
  GPUCacheManager(Context ctx, const void* all_data, DataType dtype, size_t dim,
                  const IdType* nodes, size_t num_nodes,
                  double cache_percentage);
  ~GPUCacheManager();

  void ExtractMissData(void* output_miss, IdType* output_miss_index,
                       size_t* num_output_miss, IdType* output_cache_src_index,
                       IdType* output_cache_dst_index, size_t* num_output_cache,
                       const IdType* index, const size_t num_index);
  void CombineMissData(void* output, const void* miss, const IdType* miss_index,
                       const size_t num_miss, StreamHandle stream);
  void CombineCacheData(void* output, const IdType* cache_src_index,
                        const IdType* cache_dst_index, const size_t num_cache,
                        StreamHandle stream);

 private:
  Context _ctx;
  size_t _num_nodes;
  size_t _num_cached_nodes;
  double _cache_percentage;

  DataType _dtype;
  size_t _dim;

  const void* _all_data;
  size_t _cache_nbytes;
  void* _cache_gpu_data;

  IdType* _cpu_hashtable;
};

}  // namespace cuda
}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_CUDA_CACHE_MANAGER_H
