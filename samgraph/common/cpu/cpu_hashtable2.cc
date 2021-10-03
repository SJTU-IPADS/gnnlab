#include "cpu_hashtable2.h"

#include <omp.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <unordered_set>

#include "../common.h"
#include "../device.h"
#include "../logging.h"
#include "../run_config.h"
#include "../timer.h"
#include "cpu_function.h"

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
  // 1. Populate the hashtable and count the number of insert
  std::vector<IdType> items_prefix(RunConfig::omp_thread_num + 1, 0);
  size_t task_per_worker =
      RoundUpDiv(num_input, static_cast<size_t>(RunConfig::omp_thread_num));
#pragma omp parallel num_threads(RunConfig::omp_thread_num)
  {
    IdType thread_id = omp_get_thread_num();
    size_t task_start_offset = task_per_worker * thread_id;
    size_t task_end_offset =
        std::min(task_per_worker * (thread_id + 1), num_input);
    for (size_t i = task_start_offset; i < task_end_offset; i++) {
      IdType id = input[i];
      const IdType key = __sync_val_compare_and_swap(&_o2n_table[id].key,
                                                     Constant::kEmptyKey, id);
      if (key == Constant::kEmptyKey) {
        _o2n_table[id].thread_index = thread_id;
        _o2n_table[id].version = _version;
        items_prefix[thread_id] += 1;
      }
    }
  }

  // 2. Single-thread prefix sum
  IdType prefix_sum = 0;
  for (int i = 0; i <= RunConfig::omp_thread_num; i++) {
    IdType tmp = items_prefix[i];
    items_prefix[i] = prefix_sum;
    prefix_sum += tmp;
  }

  // 3. Map old id to new id
#pragma omp parallel num_threads(RunConfig::omp_thread_num)
  {
    IdType thread_id = omp_get_thread_num();
    size_t task_start_offset = task_per_worker * thread_id;
    size_t task_end_offset =
        std::min(task_per_worker * (thread_id + 1), num_input);

    IdType next_new_id = _num_items + items_prefix[thread_id];
    for (size_t i = task_start_offset; i < task_end_offset; i++) {
      IdType id = input[i];
      BucketO2N &bucket = _o2n_table[id];
      if (bucket.thread_index == thread_id && bucket.version == _version) {
        IdType new_id = next_new_id;
        bucket.local = new_id;
        bucket.version++;  // prevent the current thread insert the value
                           // multiple times
        _n2o_table[new_id].global = id;
        next_new_id++;
      }
    }
  }

  IdType new_inserted = items_prefix[RunConfig::omp_thread_num];
  _num_items += new_inserted;
  _version += 2;  // In 3.Map old id to new id, we already increase some
                  // buckets' versions by 1
}

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