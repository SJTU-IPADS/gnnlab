#ifndef SAMGRAPH_RUN_CONFIG_H
#define SAMGRAPH_RUN_CONIFG_H

#include <string>
#include <vector>

#include "common.h"
#include "logging.h"

namespace samgraph {
namespace common {

struct RunConfig {
  // Configs passed from application
  static std::string dataset_path;
  static std::vector<int> fanout;
  static size_t batch_size;
  static size_t num_epoch;
  static Context sampler_ctx;
  static Context trainer_ctx;
  static int cpu_hashtable_type;
  static double cache_percentage;

  // Environment variables
  static bool option_profile_cuda;
  static bool option_log_node_access;
  static bool option_sanity_check;
  static bool option_stream_blocking;

  static size_t kPipelineDepth;
  static int kOMPThreadNum;

  static inline bool UseGPUCache() { return cache_percentage > 0; }
  static void LoadConfigFromEnv();
};

#define IF_BLOCKING(func)                          \
  {                                                \
    if (RunConfig::option_stream_blocking) (func); \
  }

}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_RUN_CONFIG_H