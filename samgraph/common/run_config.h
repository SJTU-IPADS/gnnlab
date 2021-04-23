#ifndef SAMGRAPH_RUN_CONFIG_H
#define SAMGRAPH_RUN_CONIFG_H

#include <string>
#include <vector>

#include "common.h"

namespace samgraph {
namespace common {

enum CpuHashTableType {
  kSimple = 0,
  kParallel,
  kOptimized,
};

struct RunConfig {
  static std::string dataset_path;
  static std::vector<int> fanout;
  static size_t batch_size;
  static size_t num_epoch;
  static Context sampler_ctx;
  static Context trainer_ctx;
  static CpuHashTableType cpu_hash_table_type;
};

}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_RUN_CONFIG_H