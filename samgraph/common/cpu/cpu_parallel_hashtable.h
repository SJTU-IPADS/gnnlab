#ifndef SAMGRAPH_CPU_PARALLEL_HASHTABLE_H
#define SAMGRAPH_CPU_PARALLEL_HASHTABLE_H

#include "../common.h"
#include "cpu_hashtable.h"

namespace samgraph {
namespace common {
namespace cpu {

class ParallelHashTable : public HashTable {
 public:
  ParallelHashTable(size_t sz);
  ~ParallelHashTable();

  void Populate(const IdType *input, const size_t num_input);
  void MapNodes(IdType *output, size_t num_output) override;
  void MapEdges(const IdType *src, const IdType *dst, const size_t len,
                IdType *new_src, IdType *new_dst) override;
  void Reset() override;
  size_t NumItems() const override { return _num_items; }

 private:
  struct Bucket0 {
    IdType id;
    IdType local;
  };

  struct Bucket1 {
    IdType global;
  };

  Bucket0 *_o2n_table;
  Bucket1 *_n2o_table;

  IdType _num_items;
  size_t _capacity;
};

}  // namespace cpu
}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_CPU_PARALLEL_HASHTABLE_H