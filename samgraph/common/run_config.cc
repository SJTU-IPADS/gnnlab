#include "run_config.h"

#include "constant.h"
#include "logging.h"

namespace samgraph {
namespace common {

std::string RunConfig::dataset_path;
RunArch RunConfig::run_arch;
SampleType RunConfig::sample_type;
std::vector<int> RunConfig::fanout;
size_t RunConfig::batch_size;
size_t RunConfig::num_epoch;
Context RunConfig::sampler_ctx;
Context RunConfig::trainer_ctx;
CachePolicy RunConfig::cache_policy;
double RunConfig::cache_percentage = 0.0f;

// CPUHash2 now is the best parallel hash remapping
cpu::CPUHashType RunConfig::cpu_hash_type = cpu::kCPUHash2;

bool RunConfig::option_report_step = false;
bool RunConfig::option_profile_cuda = false;
bool RunConfig::option_log_node_access = false;
bool RunConfig::option_sanity_check = false;

size_t RunConfig::kPipelineDepth = 10;
int RunConfig::kOMPThreadNum = 24;

void RunConfig::LoadConfigFromEnv() {
  if (IsEnvSet(Constant::kEnvReportStep)) {
    RunConfig::option_report_step = true;
  }

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