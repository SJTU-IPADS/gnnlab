#pragma once

#include "../common.h"
#include <string>
#include <vector>

namespace samgraph {
namespace common {
namespace coll_cache {

/**
 * @brief Describe asymmetric link topo at global view
 */
struct AsymmLinkDesc {

  enum TopoType {
    kHardWiredSymm,
    kHardWiredAsymm,
    kSwitch,
  };
  
  TopoType _topo_type;
  template<typename T>
  using vec = std::vector<T>;
  vec<vec<vec<int>>> link_src;
  vec<vec<double>> link_time;
  vec<vec<double>> compute_percent;
  vec<vec<int>> link_sm;
  vec<double> aggregated_time;
  AsymmLinkDesc() {}
  void BuildSwitch(int num_trainer);
  void BuildSymmHardWire(int num_trainer);
  void BuildAsymmHardWire(int num_trainer);
  static AsymmLinkDesc AutoBuild(int num_trainer, int total_gpu, std::string gpu_model);
  static AsymmLinkDesc AutoBuild(Context ctx);
  void SMPercentToNum(int total_sm);
  double AggregatedRemoteTime();
};

bool AutoEnableConcurrentLink();
}
}
}