#include "cpu_hashtable0.h"

#include "../device.h"
#include "../logging.h"

namespace samgraph {
namespace common {
namespace cpu {

CPUHashTable0::CPUHashTable0(size_t max_items) {
  _n2o_table = static_cast<BucketN2O *>(
      Device::Get(CPU())->AllocDataSpace(CPU(), max_items * sizeof(BucketN2O)));
  _num_items = 0;
}

CPUHashTable0::~CPUHashTable0() {
  Device::Get(CPU())->FreeDataSpace(CPU(), _n2o_table);
}

void CPUHashTable0::Populate(const IdType *input, const size_t num_input) {
  for (size_t i = 0; i < num_input; i++) {
    IdType oid = input[i];
    IdType nid = _num_items;
    auto res = _o2n_table.insert({oid, nid});
    if (res.second) {
      _n2o_table[nid].global = oid;
      _num_items++;
    }
  }
}

void CPUHashTable0::MapNodes(IdType *output, size_t num_output) {
  memcpy(output, _n2o_table, sizeof(IdType) * num_output);
}

void CPUHashTable0::MapEdges(const IdType *src, const IdType *dst,
                             const size_t len, IdType *new_src,
                             IdType *new_dst) {
  for (size_t i = 0; i < len; i++) {
    auto it0 = _o2n_table.find(src[i]);
    auto it1 = _o2n_table.find(dst[i]);

    CHECK(it0 != _o2n_table.end());
    CHECK(it1 != _o2n_table.end());

    new_src[i] = it0->second;
    new_dst[i] = it1->second;
  }
}

void CPUHashTable0::Reset() {
  _o2n_table = phmap::flat_hash_map<IdType, IdType>();
  _num_items = 0;
}

}  // namespace cpu
}  // namespace common
}  // namespace samgraph
