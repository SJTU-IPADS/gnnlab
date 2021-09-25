#include "cpu_hashtable1.h"

#include <cstdlib>
#include <cstring>

#include "../common.h"
#include "../device.h"
#include "../logging.h"
#include "../run_config.h"
#include "../timer.h"

namespace samgraph {
namespace common {
namespace cpu {

CPUHashTable1::CPUHashTable1(size_t max_items) {
  _o2n_table = static_cast<BucketO2N *>(
      Device::Get(CPU())->AllocDataSpace(CPU(), max_items * sizeof(BucketO2N)));
  _n2o_table = static_cast<BucketN2O *>(
      Device::Get(CPU())->AllocDataSpace(CPU(), max_items * sizeof(BucketN2O)));

  _num_items = 0;
  _capacity = max_items;

  InitTable();
}

CPUHashTable1::~CPUHashTable1() {
  Device::Get(CPU())->FreeDataSpace(CPU(), _o2n_table);
  Device::Get(CPU())->FreeDataSpace(CPU(), _n2o_table);
}

void CPUHashTable1::Populate(const IdType *input, const size_t num_input) {
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for (size_t i = 0; i < num_input; i++) {
    IdType id = input[i];
    const IdType key = __sync_val_compare_and_swap(&_o2n_table[id].id,
                                                   Constant::kEmptyKey, id);
    if (key == Constant::kEmptyKey) {
      IdType local = __sync_fetch_and_add(&_num_items, 1);
      _o2n_table[id].local = local;
      _n2o_table[local].global = id;
    }
  }
}

void CPUHashTable1::MapNodes(IdType *output, size_t num_ouput) {
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for (size_t i = 0; i < num_ouput; i++) {
    output[i] = _n2o_table[i].global;
  }
}

void CPUHashTable1::MapEdges(const IdType *src, const IdType *dst,
                             const size_t len, IdType *new_src,
                             IdType *new_dst) {
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for (size_t i = 0; i < len; i++) {
    BucketO2N &bucket0 = _o2n_table[src[i]];
    BucketO2N &bucket1 = _o2n_table[dst[i]];

    new_src[i] = bucket0.local;
    new_dst[i] = bucket1.local;
  }
}

void CPUHashTable1::Reset() {
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for (size_t i = 0; i < _num_items; i++) {
    IdType key = _n2o_table[i].global;
    _o2n_table[key].id = Constant::kEmptyKey;
  }
  _num_items = 0;
}

void CPUHashTable1::InitTable() {
  _num_items = 0;
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for (size_t i = 0; i < _capacity; i++) {
    _o2n_table[i].id = Constant::kEmptyKey;
  }
}

}  // namespace cpu
}  // namespace common
}  // namespace samgraph
