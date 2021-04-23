#ifndef SAMGRAPH_CPU_HASHTABLE_H
#define SAMGRAPH_CPU_HASHTABLE_H

#include "../common.h"

namespace samgraph {
namespace common {
namespace cpu {

class HashTable {
 public:
  virtual ~HashTable() {}
  virtual void Populate(const IdType *input, const size_t num_input) = 0;
  virtual void MapNodes(IdType *ouput, size_t num_output) = 0;
  virtual void MapEdges(const IdType *src, const IdType *dst, const size_t len,
                        IdType *new_src, IdType *new_dst) = 0;
  virtual void Reset() = 0;
  virtual size_t NumItems() const = 0;
};

}  // namespace cpu
}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_CPU_HASHTABLE_H