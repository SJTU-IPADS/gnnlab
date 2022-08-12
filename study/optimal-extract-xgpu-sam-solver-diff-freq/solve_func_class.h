#pragma once
#include "common/coll_cache/optimal_solver_class.h"
#include "common/common.h"
#include "common/run_config.h"
#include <vector>

namespace samgraph {
namespace common {
namespace coll_cache {

class RepSolver : public SingleStreamSolverBase {
public:
  void Solve(std::vector<int> device_to_stream,
             std::vector<int> device_to_cache_percent, std::string mode,
             double T_local = 2, double T_remote = 15, double T_cpu = 60);
};

class MultiStreamSolverBase : public CollCacheSolver {
public:
  void Build(TensorPtr stream_id_list, TensorPtr stream_freq_list,
             std::vector<int> device_to_stream,
             const IdType num_node,
             const TensorPtr nid_to_block_tensor) override;
  TensorPtr stream_id_list;
  TensorPtr stream_freq_list;
  TensorPtr nid_to_block;
  TensorPtr nid_to_rank_tensor;
};

class SelfishSolver : public MultiStreamSolverBase {
public:
  void Solve(std::vector<int> device_to_stream,
             std::vector<int> device_to_cache_percent, std::string mode,
             double T_local = 2, double T_remote = 15, double T_cpu = 60);
};

class PartRepMultiStream : public MultiStreamSolverBase {
public:
  void Build(TensorPtr stream_id_list, TensorPtr stream_freq_list,
             std::vector<int> device_to_stream,
             const IdType num_node,
             const TensorPtr nid_to_block_tensor) override;
  void Solve(std::vector<int> device_to_stream,
             std::vector<int> device_to_cache_percent, std::string mode,
             double T_local = 2, double T_remote = 15, double T_cpu = 60) override;
  TensorPtr global_rank_to_freq;
  TensorPtr global_rank_to_nid;
};

class PartitionMultiStream : public PartRepMultiStream {
public:
  void Solve(std::vector<int> device_to_stream,
             std::vector<int> device_to_cache_percent, std::string mode,
             double T_local = 2, double T_remote = 15, double T_cpu = 60) override;
};

class CliqueGlobalFreqSolver : public PartRepMultiStream {
 public:
  CliqueGlobalFreqSolver(int clique_size) : clique_size(clique_size) {}
  void Solve(std::vector<int> device_to_stream,
             std::vector<int> device_to_cache_percent, std::string mode,
             double T_local = 2, double T_remote = 15,
             double T_cpu = 60) override;
  int clique_size;
};
class CliqueLocalFreqSolver : public PartRepMultiStream {
 public:
  CliqueLocalFreqSolver(int clique_size) : clique_size(clique_size) {}
  void Build(TensorPtr stream_id_list, TensorPtr stream_freq_list,
             std::vector<int> device_to_stream, const IdType num_node,
             const TensorPtr nid_to_block_tensor) override;
  void Solve(std::vector<int> device_to_stream,
             std::vector<int> device_to_cache_percent, std::string mode,
             double T_local = 2, double T_remote = 15,
             double T_cpu = 60) override;
  int clique_size;
};

void solve_local_intuitive(TensorPtr stream_id_list, TensorPtr stream_freq_list,
                           const IdType num_node,
                           std::vector<int> device_to_stream,
                           std::vector<int> /*device_to_cache_percent*/ __,
                           TensorPtr nid_to_block, TensorPtr &block_placement,
                           std::string _, double T_local = 2,
                           double T_remote = 15, double T_cpu = 60);
} // namespace coll_cache
} // namespace common
} // namespace samgraph