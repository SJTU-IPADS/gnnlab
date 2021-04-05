#ifndef SAMGRAPH_CPU_HASHTABLE_H
#define SAMGRAPH_CPU_HASHTABLE_H

#include "../common.h"

namespace samgraph {
namespace common {
namespace cpu {

class HashTable {
 public:
  struct Bucket {
    IdType local;
    IdType version;
  };

  struct Mapping {
    IdType global;
  };

  HashTable(size_t sz);
  ~HashTable();

  void Populate(const IdType *input, const size_t num_input);
  void MapNodes(IdType *output, size_t num_output);
  void MapEdges(const IdType *src, const IdType *dst, const size_t len, IdType *new_src, IdType *new_dst);
  void Clear();
  size_t NumItem() const { return _map_offset; }

 private:
  Bucket *_table;
  Mapping *_mapping;
  
  IdType _map_offset;
  size_t _capacity;

  IdType _version;
};

} // namespace cpu
} // namespace common
} // namespace samgraph

#endif // SAMGRAPH_CPU_HASHTABLE_H