#ifndef SAMGRAPH_RUN_CONFIG_H
#define SAMGRAPH_RUN_CONIFG_H

#include <string>
#include <vector>

#include "common.h"
#include "cpu/cpu_hashtable.h"

namespace samgraph {
namespace common {

struct RunConfig {
  static std::string dataset_path;
  static std::vector<int> fanout;
  static size_t batch_size;
  static size_t num_epoch;
  static Context sampler_ctx;
  static Context trainer_ctx;

  static cpu::HashTableType cpu_hashtable_type;

  // Environment variables
  static bool option_profile_cuda;
  static bool option_log_node_access;
  static bool option_sanity_check;

  static size_t kPipelineDepth;
  static int kOMPThreadNum;

  static void LoadConfigFromEnv();
};

}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_RUN_CONFIG_H