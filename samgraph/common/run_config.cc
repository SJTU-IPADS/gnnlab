/*
 * Copyright 2022 Institute of Parallel and Distributed Systems, Shanghai Jiao Tong University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include "run_config.h"

#include "constant.h"
#include "logging.h"

namespace samgraph {
namespace common {

std::unordered_map<std::string, std::string> RunConfig::raw_configs;

// clang-format off
std::string          RunConfig::dataset_path;
RunArch              RunConfig::run_arch;
SampleType           RunConfig::sample_type;
size_t               RunConfig::batch_size;
size_t               RunConfig::num_epoch;
size_t               RunConfig::step_max_boundary              = std::numeric_limits<size_t>::max();
Context              RunConfig::sampler_ctx;
Context              RunConfig::trainer_ctx;
CachePolicy          RunConfig::cache_policy;
double               RunConfig::cache_percentage               = 0.0f;
bool                 RunConfig::unsupervised_sample            = false;
size_t               RunConfig::negative_sample_K              = 1;
NegativeSampleType   RunConfig::negative_sample_type           = NegativeSampleType::kUniform;
bool                 RunConfig::negative_sample_reuse_src      = true;

size_t               RunConfig::max_sampling_jobs              = 10;
size_t               RunConfig::max_copying_jobs               = 10;

std::vector<size_t>  RunConfig::fanout;
size_t               RunConfig::random_walk_length;
double               RunConfig::random_walk_restart_prob;
size_t               RunConfig::num_random_walk;
size_t               RunConfig::num_neighbor;
size_t               RunConfig::num_layer;

size_t               RunConfig::hiddem_dim                     = 256;
double               RunConfig::lr                             = 0.003;
double               RunConfig::dropout                        = 0.5;

bool                 RunConfig::is_configured                  = false;

// CPUHash2 now is the best parallel hash remapping
cpu::CPUHashType     RunConfig::cpu_hash_type                  = cpu::kCPUHash2;

size_t               RunConfig::num_sample_worker;
size_t               RunConfig::num_train_worker;
bool                 RunConfig::have_switcher                  = false;

// For arch7
size_t               RunConfig::worker_id                      = false;
size_t               RunConfig::num_worker                     = false;

bool                 RunConfig::option_profile_cuda            = false;
bool                 RunConfig::option_log_node_access         = false;
bool                 RunConfig::option_log_node_access_simple  = false;
bool                 RunConfig::option_sanity_check            = false;
bool                 RunConfig::option_samback_cuda_launch_blocking = false;

// env key: on -1, all epochs; on 0: no barrier; on other: which epoch to barrier
int                  RunConfig::barriered_epoch;
int                  RunConfig::presample_epoch;
bool                 RunConfig::option_dump_trace              = false;
size_t               RunConfig::option_empty_feat              = 0;

std::string          RunConfig::option_train_set_slice_mode = "";
double               RunConfig::option_train_set_percent = 100;
int                  RunConfig::option_train_set_part_num = 1;
int                  RunConfig::option_train_set_part_idx = 0;

size_t               RunConfig::option_fake_feat_dim = 0;

int                  RunConfig::omp_thread_num                 = 40;

std::string          RunConfig::shared_meta_path               = "/shared_meta_data";
// clang-format on

bool                 RunConfig::unified_memory                 = false;
std::vector<double>  RunConfig::unified_memory_percentages;
UMPolicy             RunConfig::unified_memory_policy          = UMPolicy::kDefault;
std::vector<Context> RunConfig::unified_memory_ctxes;

bool                 RunConfig::coll_cache_no_group    = false;
size_t               RunConfig::coll_cache_num_slot    = 100;
double               RunConfig::coll_cache_coefficient = 1.05;
double               RunConfig::coll_cache_hyperparam_T_local  = 1;
double               RunConfig::coll_cache_hyperparam_T_remote = 438 / (double)213;  // performance on A100
double               RunConfig::coll_cache_hyperparam_T_cpu    = 438 / (double)11.8; // performance on A100

void RunConfig::LoadConfigFromEnv() {
  if (IsEnvSet(Constant::kEnvSamBackCudaLaunchBlocking)) {
    RunConfig::option_samback_cuda_launch_blocking = true;
  }
  if (IsEnvSet(Constant::kEnvProfileCuda)) {
    RunConfig::option_profile_cuda = true;
  }

  if (IsEnvSet(Constant::kEnvLogNodeAccessSimple)) {
    RunConfig::option_log_node_access_simple = true;
  }

  if (IsEnvSet(Constant::kEnvLogNodeAccess)) {
    RunConfig::option_log_node_access = true;
  }

  if (IsEnvSet(Constant::kEnvSanityCheck)) {
    RunConfig::option_sanity_check = true;
  }

  if (IsEnvSet(Constant::kEnvDumpTrace)) {
    RunConfig::option_dump_trace = true;
  }
  if (GetEnv(Constant::kEnvEmptyFeat) != "") {
    RunConfig::option_empty_feat = std::stoul(GetEnv(Constant::kEnvEmptyFeat));
  }

  if (GetEnv(Constant::kEnvTrainSetPart) != "") {
    std::string env = GetEnv(Constant::kEnvTrainSetPart);
    size_t div = env.find('/');
    if (div == env.npos) {
      // env looks like "train_set_percent
      RunConfig::option_train_set_slice_mode = "percent";
      RunConfig::option_train_set_percent = std::stod(env);
    } else {
      // env looks like "part_idx/part_num"
      RunConfig::option_train_set_slice_mode = "part";
      RunConfig::option_train_set_part_num = std::stod(env.substr(div + 1));
      RunConfig::option_train_set_part_idx = std::stoi(env.substr(0, div));
    }
  }

  if (GetEnv(Constant::kEnvFakeFeatDim) != "") {
    std::string env = GetEnv(Constant::kEnvFakeFeatDim);
    RunConfig::option_fake_feat_dim = std::stoi(env);
  }
  if (IsEnvSet("SAMGRAPH_COLL_CACHE_NO_GROUP")) {
    RunConfig::coll_cache_no_group = true;
  }
}


}  // namespace common
}  // namespace samgraph
