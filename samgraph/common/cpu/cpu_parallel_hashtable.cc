#include "cpu_parallel_hashtable.h"

#include <cstdlib>
#include <cstring>

#include "../config.h"
#include "../logging.h"
#include "../timer.h"

namespace samgraph {
namespace common {
namespace cpu {

ParallelHashTable::ParallelHashTable(size_t sz) {
  _table = (Bucket *)malloc(sz * sizeof(Bucket));
  _mapping = (Mapping *)malloc(sz * sizeof(Mapping));

  _map_offset = 0;
  _capacity = sz;
}

ParallelHashTable::~ParallelHashTable() {
  free(_table);
  free(_mapping);
}

void ParallelHashTable::Populate(const IdType *input, const size_t num_input) {
#pragma omp parallel for num_threads(Config::kOmpThreadNum)
  for (size_t i = 0; i < num_input; i++) {
    IdType id = input[i];
    SAM_CHECK_LT(id, _capacity);
    const IdType key =
        __sync_val_compare_and_swap(&_table[id].id, Config::kEmptyKey, id);
    if (key == Config::kEmptyKey) {
      IdType local = __sync_fetch_and_add(&_map_offset, 1);
      _table[id].local = local;
      _mapping[local].global = id;
    }
  }
}

void ParallelHashTable::MapNodes(IdType *output, size_t num_ouput) {
  SAM_CHECK_LE(num_ouput, _map_offset);
#pragma omp parallel for num_threads(Config::kOmpThreadNum)
  for (size_t i = 0; i < num_ouput; i++) {
    output[i] = _mapping[i].global;
  }
}

void ParallelHashTable::MapEdges(const IdType *src, const IdType *dst,
                                 const size_t len, IdType *new_src,
                                 IdType *new_dst) {
#pragma omp parallel for num_threads(Config::kOmpThreadNum)
  for (size_t i = 0; i < len; i++) {
    SAM_CHECK_LT(src[i], _capacity);
    SAM_CHECK_LT(dst[i], _capacity);

    Bucket &bucket0 = _table[src[i]];
    Bucket &bucket1 = _table[dst[i]];

    new_src[i] = bucket0.local;
    new_dst[i] = bucket1.local;
  }
}

void ParallelHashTable::Clear() {
  _map_offset = 0;
#pragma omp parallel for num_threads(Config::kOmpThreadNum)
  for (size_t i = 0; i < _capacity; i++) {
    _table[i].id = Config::kEmptyKey;
  }
}

}  // namespace cpu
}  // namespace common
}  // namespace samgraph
