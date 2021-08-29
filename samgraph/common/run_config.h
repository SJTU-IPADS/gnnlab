#ifndef SAMGRAPH_RUN_CONFIG_H
#define SAMGRAPH_RUN_CONFIG_H

#include <string>
#include <vector>

#include "common.h"
#include "cpu/cpu_common.h"
// #include "logging.h"

namespace samgraph {
namespace common {

struct RunConfig {
  // Configs passed from application
  static std::string dataset_path;
  static RunArch run_arch;
  static SampleType sample_type;
  static size_t batch_size;
  static size_t num_epoch;
  static Context sampler_ctx;
  static Context trainer_ctx;
  static CachePolicy cache_policy;
  static double cache_percentage;

  static size_t max_sampling_jobs;
  static size_t max_copying_jobs;

  static std::vector<size_t> fanout;
  static size_t random_walk_length;
  static double random_walk_restart_prob;
  static size_t num_random_walk;
  static size_t num_neighbor;
  static size_t num_layer;

  static bool is_configured;
  static bool is_khop_configured;
  static bool is_random_walk_configured;

  static cpu::CPUHashType cpu_hash_type;

  // Environment variables
  static bool option_profile_cuda;
  static bool option_log_node_access;
  static bool option_sanity_check;
  static int barriered_epoch;
  static bool option_dump_trace;

  static int kOMPThreadNum;

  static inline bool UseGPUCache() {
    return cache_percentage > 0 && run_arch != kArch1;
  }
  static inline bool UseDynamicGPUCache() {
    return cache_policy == kDynamicCache;
  }
  static void LoadConfigFromEnv();
};

}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_RUN_CONFIG_H