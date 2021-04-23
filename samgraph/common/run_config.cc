#include "run_config.h"

namespace samgraph {
namespace common {

std::string RunConfig::dataset_path;
std::vector<int> RunConfig::fanout;
size_t RunConfig::batch_size;
size_t RunConfig::num_epoch;
Context RunConfig::sampler_ctx;
Context RunConfig::trainer_ctx;
cpu::HashTableType RunConfig::cpu_hashtable_type;

}  // namespace common
}  // namespace samgraph