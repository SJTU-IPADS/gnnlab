#include "cpu_hashtable2.h"

#include <omp.h>

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

CPUHashTable2::CPUHashTable2(size_t max_items) {
  _o2n_table = static_cast<BucketO2N *>(
      Device::Get(CPU())->AllocDataSpace(CPU(), max_items * sizeof(BucketO2N)));
  _n2o_table = static_cast<BucketN2O *>(
      Device::Get(CPU())->AllocDataSpace(CPU(), max_items * sizeof(BucketN2O)));

  _num_items = 0;
  _version = 0;
  _capacity = max_items;

  InitTable();
}

CPUHashTable2::~CPUHashTable2() {
  Device::Get(CPU())->FreeDataSpace(CPU(), _o2n_table);
  Device::Get(CPU())->FreeDataSpace(CPU(), _n2o_table);
}

void CPUHashTable2::Populate(const IdType *input, const size_t num_input) {
  // 1. Populate the hashtable
  // Note: OMP should use static schedule
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for (size_t i = 0; i < num_input; i++) {
    IdType id = input[i];
    const IdType key = __sync_val_compare_and_swap(&_o2n_table[id].key,
                                                   Constant::kEmptyKey, id);
    if (key == Constant::kEmptyKey) {
      _o2n_table[id].index = i;
      _o2n_table[id].version = _version;
    }
  }

  // 2. Count the number of insert
  // Note: OMP should use static schedule
  std::vector<PrefixItem> items_prefix(RunConfig::omp_thread_num + 1);
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for (size_t i = 0; i < num_input; i++) {
    IdType id = input[i];
    IdType my_thread_idx = omp_get_thread_num();
    const BucketO2N &bucket = _o2n_table[id];
    if (bucket.index == i && bucket.version == _version) {
      items_prefix[my_thread_idx].val++;
    }
  }

  // 3. Single-thread prefix sum
  size_t prefix_sum = 0;
  for (int i = 0; i <= RunConfig::omp_thread_num; i++) {
    size_t tmp = items_prefix[i].val;
    items_prefix[i].val = prefix_sum;
    prefix_sum += tmp;
  }

  // 4. Map old id to new id
  // Note: OMP should use static schedule
  IdType start_off = _num_items;
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for (size_t i = 0; i < num_input; i++) {
    IdType id = input[i];
    IdType my_thread_idx = omp_get_thread_num();
    BucketO2N &bucket = _o2n_table[id];
    if (bucket.index == i && bucket.version == _version) {
      IdType new_id = start_off + items_prefix[my_thread_idx].val;
      bucket.local = new_id;
      _n2o_table[new_id].global = id;
      items_prefix[my_thread_idx].val++;
    }
  }

  IdType new_inserted = items_prefix[RunConfig::omp_thread_num].val;
  _num_items += new_inserted;
  _version++;
}

// void CPUHashTable2::Populate(const IdType *input, const size_t num_input) {
//   // 1. Populate the hashtable
//   // Note: OMP should use static schedule
// #pragma omp parallel for num_threads(RunConfig::omp_thread_num)
//   for (size_t i = 0; i < num_input; i++) {
//     IdType id = input[i];
//     const IdType key = __sync_val_compare_and_swap(&_o2n_table[id].key,
//                                                    Constant::kEmptyKey, id);
//     if (key == Constant::kEmptyKey) {
//       _o2n_table[id].index = i;
//       _o2n_table[id].version = _version;
//     }
//   }

//   // 2. Count the number of insert
//   // Note: OMP should use static schedule
//   std::vector<IdType> items_prefix(RunConfig::omp_thread_num + 1);
// #pragma omp parallel for num_threads(RunConfig::omp_thread_num)
//   for (size_t i = 0; i < num_input; i++) {
//     IdType id = input[i];
//     IdType my_thread_idx = omp_get_thread_num();
//     const BucketO2N &bucket = _o2n_table[id];
//     if (bucket.index == i && bucket.version == _version) {
//       items_prefix[my_thread_idx]++;
//     }
//   }

//   // 3. Single-thread prefix sum
//   size_t prefix_sum = 0;
//   for (int i = 0; i <= RunConfig::omp_thread_num; i++) {
//     size_t tmp = items_prefix[i];
//     items_prefix[i] = prefix_sum;
//     prefix_sum += tmp;
//   }

//   // 4. Map old id to new id
//   // Note: OMP should use static schedule
//   IdType start_off = _num_items;
// #pragma omp parallel for num_threads(RunConfig::omp_thread_num)
//   for (size_t i = 0; i < num_input; i++) {
//     IdType id = input[i];
//     IdType my_thread_idx = omp_get_thread_num();
//     BucketO2N &bucket = _o2n_table[id];
//     if (bucket.index == i && bucket.version == _version) {
//       IdType new_id = start_off + items_prefix[my_thread_idx];
//       bucket.local = new_id;
//       _n2o_table[new_id].global = id;
//       items_prefix[my_thread_idx]++;
//     }
//   }

//   IdType new_inserted = items_prefix[RunConfig::omp_thread_num];
//   _num_items += new_inserted;
//   _version++;
// }

void CPUHashTable2::MapNodes(IdType *output, size_t num_ouput) {
  CHECK_LE(num_ouput, _num_items);
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for (size_t i = 0; i < num_ouput; i++) {
    output[i] = _n2o_table[i].global;
  }
}

void CPUHashTable2::MapEdges(const IdType *src, const IdType *dst,
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

void CPUHashTable2::Reset() {
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for (size_t i = 0; i < _num_items; i++) {
    IdType key = _n2o_table[i].global;
    _o2n_table[key].key = Constant::kEmptyKey;
  }
  _num_items = 0;
  _version = 0;
}

void CPUHashTable2::InitTable() {
  _num_items = 0;
  _version = 0;
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for (size_t i = 0; i < _capacity; i++) {
    _o2n_table[i].key = Constant::kEmptyKey;
  }
}

}  // namespace cpu
}  // namespace common
}  // namespace samgraph