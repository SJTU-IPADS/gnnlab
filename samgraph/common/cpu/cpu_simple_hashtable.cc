#include "cpu_simple_hashtable.h"

#include "../device.h"
#include "../logging.h"

namespace samgraph {
namespace common {
namespace cpu {

SimpleHashTable::SimpleHashTable(size_t sz) : _filter(kFilterSize, false) {
  _n2o_table = static_cast<Bucket1 *>(
      Device::Get(CPU())->AllocDataSpace(CPU(), sz * sizeof(Bucket1)));
  _num_items = 0;
}

SimpleHashTable::~SimpleHashTable() {
  Device::Get(CPU())->FreeDataSpace(CPU(), _n2o_table);
}

void SimpleHashTable::Populate(const IdType *input, const size_t num_input) {
  for (size_t i = 0; i < num_input; i++) {
    IdType oid = input[i];
    IdType nid = _num_items;
    auto res = _o2n_table.insert({oid, nid});
    if (res.second) {
      _n2o_table[nid].global = oid;
      _filter[nid & kFilterMask] = true;
      _num_items++;
    }
  }
}

void SimpleHashTable::MapNodes(IdType *output, size_t num_output) {
  CHECK_EQ(num_output, _num_items);
  memcpy(output, _n2o_table, sizeof(IdType) * num_output);
}

void SimpleHashTable::MapEdges(const IdType *src, const IdType *dst,
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

void SimpleHashTable::Reset() {
  _o2n_table = phmap::flat_hash_map<IdType, IdType>();
  _num_items = 0;
}

}  // namespace cpu
}  // namespace common
}  // namespace samgraph
