#ifndef SAMGRAPH_CPU_PARALLEL_HASHTABLE_H
#define SAMGRAPH_CPU_PARALLEL_HASHTABLE_H

#include "../common.h"
#include "cpu_hashtable.h"

namespace samgraph {
namespace common {
namespace cpu {

class ParallelHashTable {
 public:
  struct Bucket {
    IdType local;
    IdType version;
  };

  struct Mapping {
    IdType global;
  };

  ParallelHashTable(size_t sz);
  ~ParallelHashTable();

  void Populate(const IdType *input, const size_t num_input);
  void MapNodes(IdType *output, size_t num_output);
  void MapEdges(const IdType *src, const IdType *dst, const size_t len,
                IdType *new_src, IdType *new_dst);
  void Clear();
  size_t NumItem() const { return _map_offset; }

 private:
  Bucket *_table;
  Mapping *_mapping;

  IdType _map_offset;
  size_t _capacity;

  IdType _version;
};

}  // namespace cpu
}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_CPU_PARALLEL_HASHTABLE_H