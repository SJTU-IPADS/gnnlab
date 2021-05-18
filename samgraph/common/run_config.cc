#include "run_config.h"

#include "constant.h"
#include "logging.h"

namespace samgraph {
namespace common {

std::string RunConfig::dataset_path;
std::vector<int> RunConfig::fanout;
size_t RunConfig::batch_size;
size_t RunConfig::num_epoch;
Context RunConfig::sampler_ctx;
Context RunConfig::trainer_ctx;
cpu::HashTableType RunConfig::cpu_hashtable_type;
bool RunConfig::option_profile_cuda = false;
bool RunConfig::option_log_node_access = false;

size_t RunConfig::kPipelineDepth = 5;
int RunConfig::kOMPThreadNum = 24;

void RunConfig::LoadConfigFromEnv() {
  std::string profile_cuda = GetEnv(Constant::kEnvProfileCuda);
  std::string log_node_access = GetEnv(Constant::kEnvLogNodeAccess);

  if (profile_cuda == "ON" || profile_cuda == "1") {
    LOG(INFO) << "Start cuda profiler";
    RunConfig::option_profile_cuda = true;
  }

  if (log_node_access == "ON" || log_node_access == "1") {
    LOG(INFO) << "Log node access data";
    RunConfig::option_log_node_access = true;
  }
}

}  // namespace common
}  // namespace samgraph