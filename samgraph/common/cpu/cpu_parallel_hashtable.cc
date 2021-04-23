#include "cpu_parallel_hashtable.h"

#include <cstdlib>
#include <cstring>

#include "../common.h"
#include "../config.h"
#include "../device.h"
#include "../logging.h"
#include "../timer.h"

namespace samgraph {
namespace common {
namespace cpu {

ParallelHashTable::ParallelHashTable(size_t sz) {
  _o2n_table = static_cast<Bucket0 *>(
      Device::Get(CPU())->AllocDataSpace(CPU(), sz * sizeof(Bucket0)));
  _n2o_table = static_cast<Bucket1 *>(
      Device::Get(CPU())->AllocDataSpace(CPU(), sz * sizeof(Bucket1)));

  _num_items = 0;
  _capacity = sz;
}

ParallelHashTable::~ParallelHashTable() {
  Device::Get(CPU())->FreeDataSpace(CPU(), _o2n_table);
  Device::Get(CPU())->FreeDataSpace(CPU(), _n2o_table);
}

void ParallelHashTable::Populate(const IdType *input, const size_t num_input) {
#pragma omp parallel for num_threads(Config::kOmpThreadNum)
  for (size_t i = 0; i < num_input; i++) {
    IdType id = input[i];
    CHECK_LT(id, _capacity);
    const IdType key =
        __sync_val_compare_and_swap(&_o2n_table[id].id, Config::kEmptyKey, id);
    if (key == Config::kEmptyKey) {
      IdType local = __sync_fetch_and_add(&_num_items, 1);
      _o2n_table[id].local = local;
      _n2o_table[local].global = id;
    }
  }
}

void ParallelHashTable::MapNodes(IdType *output, size_t num_ouput) {
  CHECK_LE(num_ouput, _num_items);
#pragma omp parallel for num_threads(Config::kOmpThreadNum)
  for (size_t i = 0; i < num_ouput; i++) {
    output[i] = _n2o_table[i].global;
  }
}

void ParallelHashTable::MapEdges(const IdType *src, const IdType *dst,
                                 const size_t len, IdType *new_src,
                                 IdType *new_dst) {
#pragma omp parallel for num_threads(Config::kOmpThreadNum)
  for (size_t i = 0; i < len; i++) {
    CHECK_LT(src[i], _capacity);
    CHECK_LT(dst[i], _capacity);

    Bucket0 &bucket0 = _o2n_table[src[i]];
    Bucket0 &bucket1 = _o2n_table[dst[i]];

    new_src[i] = bucket0.local;
    new_dst[i] = bucket1.local;
  }
}

void ParallelHashTable::Reset() {
  _num_items = 0;
#pragma omp parallel for num_threads(Config::kOmpThreadNum)
  for (size_t i = 0; i < _capacity; i++) {
    _o2n_table[i].id = Config::kEmptyKey;
  }
}

}  // namespace cpu
}  // namespace common
}  // namespace samgraph
