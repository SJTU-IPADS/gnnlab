#ifndef SAMGRAPH_CPU_HASHMAP_H
#define SAMGRAPH_CPU_HASHMAP_H

#include <cstdint>
#include <cstdlib>
#include <vector>

#include <unordered_map>

#include "../common.h"

namespace samgraph {
namespace common {
namespace cpu {

class HashTable {
 public:
  HashTable(size_t sz);

  void Fill(const IdType *input, const size_t num_input);
  void GetUnique(IdType *output, size_t num_output) const;
  void Map(const IdType *src, const IdType *dst, const size_t len, IdType *new_src, IdType *new_dst);
  size_t NumItems() const { return _oldv2newv.size(); }

 private:
  std::unordered_map<IdType, IdType> _oldv2newv;
};

} // namespace cpu
} // namespace common
} // namespace samgraph

#endif // SAMGRAPH_CPU_HASHMAP_H