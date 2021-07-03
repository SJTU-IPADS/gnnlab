#include "run_config.h"

#include "constant.h"
#include "logging.h"

namespace samgraph {
namespace common {

std::string RunConfig::dataset_path;
RunArch RunConfig::run_arch;
SampleType RunConfig::sample_type;
size_t RunConfig::batch_size;
size_t RunConfig::num_epoch;
Context RunConfig::sampler_ctx;
Context RunConfig::trainer_ctx;
CachePolicy RunConfig::cache_policy;
double RunConfig::cache_percentage = 0.0f;

size_t RunConfig::max_sampling_jobs = 10;
size_t RunConfig::max_copying_jobs = 10;

std::vector<size_t> RunConfig::fanout;
size_t RunConfig::random_walk_length;
double RunConfig::random_walk_restart_prob;
size_t RunConfig::num_random_walk;
size_t RunConfig::num_neighbor;
size_t RunConfig::num_layer;

bool RunConfig::is_configured = false;
bool RunConfig::is_khop_configured = false;
bool RunConfig::is_random_walk_configured = false;

// CPUHash2 now is the best parallel hash remapping
cpu::CPUHashType RunConfig::cpu_hash_type = cpu::kCPUHash2;

bool RunConfig::option_profile_cuda = false;
bool RunConfig::option_log_node_access = false;
bool RunConfig::option_sanity_check = false;

int RunConfig::kOMPThreadNum = 40;

void RunConfig::LoadConfigFromEnv() {
  if (IsEnvSet(Constant::kEnvProfileCuda)) {
    RunConfig::option_profile_cuda = true;
  }

  if (IsEnvSet(Constant::kEnvLogNodeAccess)) {
    RunConfig::option_log_node_access = true;
  }

  if (IsEnvSet(Constant::kEnvSanityCheck)) {
    RunConfig::option_sanity_check = true;
  }
}

}  // namespace common
}  // namespace samgraph