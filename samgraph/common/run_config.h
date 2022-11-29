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

#ifndef SAMGRAPH_RUN_CONFIG_H
#define SAMGRAPH_RUN_CONFIG_H

#include <string>
#include <unordered_map>
#include <vector>

#include "common.h"
#include "cpu/cpu_common.h"
// #include "coll_cache/asymm_link_desc.h"

namespace samgraph {
namespace common {

struct RunConfig {
  static std::unordered_map<std::string, std::string> raw_configs;

  // Configs passed from application
  // clang-format off
  static std::string          dataset_path;
  static RunArch              run_arch;
  static SampleType           sample_type;
  static size_t               batch_size;
  static size_t               num_epoch;
  static size_t               step_max_boundary;
  static Context              sampler_ctx;
  static Context              trainer_ctx;
  static CachePolicy          cache_policy;
  static double               cache_percentage;
  static bool                 unsupervised_sample;
  static size_t               negative_sample_K;
  static NegativeSampleType   negative_sample_type;
  static bool                 negative_sample_reuse_src;

  static size_t               max_sampling_jobs;
  static size_t               max_copying_jobs;

  static std::vector<size_t>  fanout;
  static size_t               random_walk_length;
  static double               random_walk_restart_prob;
  static size_t               num_random_walk;
  static size_t               num_neighbor;
  static size_t               num_layer;

  // model parameters
  static size_t               hiddem_dim;
  static double               dropout;
  static double               lr;

  static bool                 is_configured;

  static cpu::CPUHashType     cpu_hash_type;

  // For multi-gpu sampling and training
  static size_t               num_sample_worker;
  static size_t               num_train_worker;
  // If transform the input_nodes for Switcher architecture
  static bool                 have_switcher;
  // For arch7
  static size_t               worker_id;
  static size_t               num_worker;

  // Environment variables
  static bool                 option_profile_cuda;
  static bool                 option_log_node_access;
  static bool                 option_log_node_access_simple;
  static bool                 option_sanity_check;
  static bool                 option_samback_cuda_launch_blocking;
  static int                  barriered_epoch;
  static int                  presample_epoch;
  static bool                 option_dump_trace;
  static size_t               option_empty_feat;
  static size_t               option_mq_size;

  static std::string          option_train_set_slice_mode;
  static double               option_train_set_percent;
  static int                  option_train_set_part_num;
  static int                  option_train_set_part_idx;
  
  static size_t               option_fake_feat_dim;

  static int                  omp_thread_num;

  // shared memory meta_data path for data communication acrossing processes
  static std::string          shared_meta_path;
  // clang-format on

  static bool                 unified_memory;
  // how much percentage of unified_memory data to be stored in GPU
  // note: unified_memory_percentage is in the range [0, 1]
  static std::vector<double>  unified_memory_percentages;
  static UMPolicy             unified_memory_policy;
  static std::vector<Context> unified_memory_ctxes;

  static bool                 coll_cache_concurrent_link;
  static bool                 coll_cache_no_group;
  static size_t               coll_cache_num_slot;
  static double               coll_cache_coefficient;
  static double               coll_cache_hyperparam_T_local;
  static double               coll_cache_hyperparam_T_remote;
  static double               coll_cache_hyperparam_T_cpu;

  // static coll_cache::AsymmLinkDesc coll_cache_link_desc;

  static RollingPolicy        rolling;

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
