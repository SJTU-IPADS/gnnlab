#ifndef SAMGRAPH_CPU_HASHTABLE1_H
#define SAMGRAPH_CPU_HASHTABLE1_H

#include "../common.h"
#include "cpu_hashtable.h"

namespace samgraph {
namespace common {
namespace cpu {

// A simple multi-thread parallel hashtable
class CPUHashTable1 : public CPUHashTable {
 public:
  CPUHashTable1(size_t max_items);
  ~CPUHashTable1();

  void Populate(const IdType *input, const size_t num_input);
  void MapNodes(IdType *output, size_t num_output) override;
  void MapEdges(const IdType *src, const IdType *dst, const size_t len,
                IdType *new_src, IdType *new_dst) override;
  void Reset() override;
  size_t NumItems() const override { return _num_items; }

 private:
  struct BucketO2N {
    IdType id;
    IdType local;
  };

  struct BucketN2O {
    IdType global;
  };

  BucketO2N *_o2n_table;
  BucketN2O *_n2o_table;

  IdType _num_items;
  size_t _capacity;

  void InitTable();
};

}  // namespace cpu
}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_CPU_HASHTABLE1_H