#include "asymm_link_desc.h"
#include "../run_config.h"
#include <iostream>
#include <cuda_runtime.h>

#define FOR_LOOP(iter, len) for (uint32_t iter = 0; iter < (len); iter++)
#define FOR_LOOP_1(iter, len) for (uint32_t iter = 1; iter < (len); iter++)

namespace samgraph {
namespace common {
namespace coll_cache {

void AsymmLinkDesc::BuildSwitch(int num_trainer) {
  _topo_type = kSwitch;
  link_src = vec<vec<vec<int>>>(num_trainer, vec<vec<int>>(1, vec<int>(num_trainer - 1)));
  for (int dst_dev = 0; dst_dev < num_trainer; dst_dev++) {
    // each device has single link, which connect to all remote device
    std::cout << dst_dev << " : link #0 : ";
    for (int remote_device = 0; remote_device < num_trainer - 1;
         remote_device++) {
      link_src[dst_dev][0][remote_device] = {(dst_dev + remote_device + 1) %  num_trainer};
      std::cout << link_src[dst_dev][0][remote_device] << ",";
    }
    std::cout << "\n";
  }
  link_time = vec<vec<double>>(num_trainer, vec<double>(1, RunConfig::coll_cache_hyperparam_T_remote));
  compute_percent = vec<vec<double>>(num_trainer, vec<double>(1, 1.));
}
void AsymmLinkDesc::BuildSymmHardWire(int num_trainer) {
  _topo_type = kHardWiredSymm;
  int num_link = num_trainer - 1;
  link_src = vec<vec<vec<int>>>(num_trainer, vec<vec<int>>(num_link));
  for (int dst_dev = 0; dst_dev < num_trainer; dst_dev++) {
    // each device has multiple link, each link contains only one remote device
    for (int src_link = 0; src_link < num_link; src_link++) {
      link_src[dst_dev][src_link] = {(dst_dev + src_link + 1) % num_trainer};
      std::cout << dst_dev << " : link #" << src_link << " : ";
      std::cout << link_src[dst_dev][src_link][0] << "\n";
    }
  }
  link_time = vec<vec<double>>(
      num_trainer,
      vec<double>(num_link, RunConfig::coll_cache_hyperparam_T_remote));
  compute_percent =
      vec<vec<double>>(num_trainer, vec<double>(num_link, 1. / num_link));
}
void AsymmLinkDesc::BuildAsymmHardWire(int num_trainer) {
  _topo_type = kHardWiredAsymm;
  double fast_link_time = RunConfig::coll_cache_hyperparam_T_remote;
  double slow_link_time = RunConfig::coll_cache_hyperparam_T_remote * 2;
  if (num_trainer == 8) {
    link_src = {
        /**
         * ┌-----------┐
         * 2 = 1 = 7 = 4
         * ║ x │   │ x ║
         * 3 = 0 = 6 = 5
         * └-----------┘
         */
        {{3}, {6}, {1}, {2}}, // 0
        {{7}, {2}, {3}, {0}}, // 1
        {{1}, {3}, {0}, {4}}, // 2
        {{2}, {0}, {5}, {1}}, // 3
        {{5}, {7}, {2}, {6}}, // 4
        {{6}, {4}, {7}, {3}}, // 5
        {{0}, {5}, {4}, {7}}, // 6
        {{4}, {1}, {6}, {5}}, // 7
    };
    link_time = vec<vec<double>>(num_trainer, 
        {{fast_link_time, fast_link_time, slow_link_time, slow_link_time},});
    compute_percent = vec<vec<double>>(
        num_trainer, vec<double>({1./3, 1./3, 1./6, 1./6}));
    return;
  }
  if (num_trainer == 6) {
    // remove 0 & 6 by reordering CUDA_VISIBLE_DEVICES
    link_src = {
        /**
         * ┌-----------┐
         * 1 = 0 = 5 = 3
         * ║ ╱       ╲ ║
         * 2           4
         * └-----------┘
         */
        // {{1}, {5}, {2}, { }}, // 0
        // {{0}, {2}, {3}, { }}, // 1
        // {{1}, { }, {0}, {4}}, // 2
        // {{4}, {5}, {1}, { }}, // 3
        // {{3}, { }, {2}, {5}}, // 4
        // {{0}, {3}, {4}, { }}, // 5
        {{1}, {5}, {2},    }, // 0
        {{0}, {2}, {3},    }, // 1
        {{1},      {0}, {4}}, // 2
        {{4}, {5}, {1},    }, // 3
        {{3},      {2}, {5}}, // 4
        {{0}, {3}, {4},    }, // 5
    };
    // link_time = vec<vec<double>>(num_trainer,{{fast_link_time, fast_link_time, slow_link_time, slow_link_time},});
    link_time = {
        {fast_link_time, fast_link_time, slow_link_time},
        {fast_link_time, fast_link_time, slow_link_time},
        {fast_link_time,                 slow_link_time, slow_link_time},
        {fast_link_time, fast_link_time, slow_link_time},
        {fast_link_time,                 slow_link_time, slow_link_time},
        {fast_link_time, fast_link_time, slow_link_time},
    };
    compute_percent = {
        // {2./5, 2./5, 1./5,    0},
        // {2./5, 2./5, 1./5,    0},
        // {1./2,    0, 1./4, 1./4},
        // {2./5, 2./5, 1./5,    0},
        // {1./2,    0, 1./4, 1./4},
        // {2./5, 2./5, 1./5,    0},
        {2./5, 2./5, 1./5,     },
        {2./5, 2./5, 1./5,     },
        {1./2,       1./4, 1./4},
        {2./5, 2./5, 1./5,     },
        {1./2,       1./4, 1./4},
        {2./5, 2./5, 1./5,     },
    };
    return;
  }
  CHECK(false) << "Unsupported configuration";
}

AsymmLinkDesc AsymmLinkDesc::AutoBuild(int num_trainer, int total_gpu,
                                       std::string gpu_model) {
  AsymmLinkDesc desc;
  if (total_gpu <= 4) {
    desc.BuildSymmHardWire(num_trainer);
  } else if (gpu_model.find("V100") != std::string::npos) {
    desc.BuildAsymmHardWire(num_trainer);
  } else if (gpu_model.find("A100") != std::string::npos) {
    desc.BuildSwitch(num_trainer);
  } else {
    CHECK(false) << "Unsupported configuration";
  }
  return desc;
}
AsymmLinkDesc AsymmLinkDesc::AutoBuild(Context ctx) {
  int total_gpu;
  CUDA_CALL(cudaGetDeviceCount(&total_gpu));
  cudaDeviceProp prop;
  CUDA_CALL(cudaGetDeviceProperties(&prop, ctx.device_id));
  auto desc = AutoBuild(RunConfig::num_train_worker, total_gpu, prop.name);
  desc.SMPercentToNum(prop.multiProcessorCount);
  return desc;
}
void AsymmLinkDesc::SMPercentToNum(int total_sm) {
  FOR_LOOP(dev_id, compute_percent.size()) {
    FOR_LOOP(link, compute_percent[dev_id].size()) {
      link_sm[dev_id][link] = total_sm * compute_percent[dev_id][link];
    }
  }
}

bool AutoEnableConcurrentLink() {
  switch(RunConfig::cache_policy) {
    case kCollCacheAsymmLink:
    case kCollCacheIntuitive:
    case kCollCache:
      return true;
    // case kCacheByHeuristic:
    // case kCacheByPreSample:
    // case kCacheByPreSampleStatic:
    // case kPartitionCache:
    // case kPartRepCache:
    // case kCacheByDegree:
    // case kCacheByDegreeHop:
    // case kCacheByFakeOptimal:
    // case kCacheByRandom:
    // case kDynamicCache:
    default:
      return false;
  }
}
double AsymmLinkDesc::AggregatedRemoteTime() {
  CHECK(_topo_type != kHardWiredAsymm);
  if (RunConfig::coll_cache_concurrent_link) {
    return RunConfig::coll_cache_hyperparam_T_remote / link_time[0].size();
  } else {
    return RunConfig::coll_cache_hyperparam_T_remote;
  }
}
} // namespace coll_cache
}
}