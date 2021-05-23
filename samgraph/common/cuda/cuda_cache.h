#ifndef SAMGRAPH_CUDA_CACHE_H
#define SAMGRAPH_CUDA_CACHE_H

#include "../common.h"

namespace samgraph {
namespace common {
namespace cuda {

class GPUCache {
 public:
  GPUCache(Context ctx, const void* all_data, DataType dtype, size_t dim,
           const IdType* nodes, size_t num_nodes, double cache_percentage);
  ~GPUCache();

  void ExtractMissData(void* output, IdType* output_index, size_t* num_output,
                       const IdType* index, const size_t num_index);
  void CombineCacheData(void* output, const IdType* index,
                        const size_t num_index, StreamHandle stream);
  void CombineMissData(void* output, const IdType* index,
                       const size_t num_index, StreamHandle stream);

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
  IdType* _gpu_hashtable;
};

}  // namespace cuda
}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_CUDA_CACHE_H
