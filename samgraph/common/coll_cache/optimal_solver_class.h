#pragma once
#include "../common.h"
#include "../cpu/mmap_cpu_device.h"
#include "../device.h"
#include "../logging.h"
#include "../run_config.h"
#include "gurobi_c++.h"
#include "ndarray.h"
#include <bitset>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <sys/fcntl.h>
#include <sys/mman.h>
#include <tbb/concurrent_unordered_map.h>
#include <unistd.h>
#include <vector>

#define FOR_LOOP(iter, len) for (uint32_t iter = 0; iter < (len); iter++)
#define FOR_LOOP_1(iter, len) for (uint32_t iter = 1; iter < (len); iter++)

namespace samgraph {
namespace common {
namespace coll_cache {
static_assert(sizeof(GRBVar) == sizeof(Id64Type),
              "size of GRBVar is not 8byte, cannot use tensor to hold it..");

class CollCacheSolver {
public:
  virtual ~CollCacheSolver() {}
  virtual void Build(TensorPtr stream_id_list, TensorPtr stream_freq_list,
                     std::vector<int> device_to_stream,
                     const IdType num_node,
                     const TensorPtr nid_to_block_tensor) = 0;
  virtual void Solve(std::vector<int> device_to_stream,
                     std::vector<int> device_to_cache_percent, std::string mode,
                     double T_local = 2, double T_remote = 15,
                     double T_cpu = 60) = 0;
  TensorPtr block_density_tensor;
  TensorPtr block_freq_tensor;
  TensorPtr block_placement;
};

class OptimalSolver : public CollCacheSolver {
public:
  // virtual ~OptimalSolver() {}
  void Build(TensorPtr stream_id_list, TensorPtr stream_freq_list,
             std::vector<int> device_to_stream,
             const IdType num_node,
             const TensorPtr nid_to_block_tensor) override;
  void Solve(std::vector<int> device_to_stream,
             std::vector<int> device_to_cache_percent, std::string mode,
             double T_local = 2, double T_remote = 15,
             double T_cpu = 60) override;

protected:
  /** ==================================================================
   * @brief status related to building blocks.
   *  only depends on dataset and its frequency
   *  does not depends on system configuration(i.e. cache size, num device)
   *  ==================================================================
   */

  // the frequency of node at top 1%. slot/block is sliced based on this value
  double alpha = 0;
  // how many slot to slice for each stream
  std::atomic_uint32_t next_free_block{0};
  uint32_t max_size_per_block = 10000;

  int freq_to_slot_1(float freq, uint32_t rank, IdType num_node) {
    if (freq == 0)
      return RunConfig::coll_cache_num_slot - 1;
    if (freq >= alpha)
      return 0;
    double exp = std::log2(alpha / (double)freq) / std::log2(RunConfig::coll_cache_coefficient);
    int slot = (int)std::ceil(exp);
    slot = std::min(slot, (int)RunConfig::coll_cache_num_slot - 2);
    return slot;
  }
  int freq_to_slot_2(float freq, uint32_t rank, IdType num_node) {
    return rank * (uint64_t)RunConfig::coll_cache_num_slot / num_node;
  }

  /**
   * each <slot freq> is mapped to a block, but the block may get too large.
   * so we split block into smaller blocks:
   * initially, each <slot freq> maps to a default block;
   * by inserting vertex into block, if the block exceed size limit, alloc a new
   * block (by atomic adding next_free_block) and update the <slot freq>
   * mapping.
   */
  inline IdType alloc_block() { return next_free_block.fetch_add(1); }

  struct block_identifer {
    uint32_t current_block_id = -1;
    uint32_t current_num = std::numeric_limits<uint32_t>::max();
    bool ignore_limit = false;
    std::atomic_flag latch;
    block_identifer() : latch() {}
    void set_ignore_limit() {
      while (latch.test_and_set()) {
      }
      ignore_limit = true;
      latch.clear();
    }
    uint32_t add_node(OptimalSolver *solver) {
      while (latch.test_and_set()) {
      }
      uint32_t selected_block = -1;
      if (current_num < solver->max_size_per_block || ignore_limit) {
        selected_block = current_block_id;
        current_num++;
      } else {
        current_block_id = solver->alloc_block();
        selected_block = current_block_id;
        current_num = 1;
      }
      latch.clear();
      return selected_block;
    }
  };

  struct concurrent_full_slot_map {
    const size_t _place_holder = 0xffffffffffffffff;
    std::atomic_uint32_t next_free_slot{0};
    tbb::concurrent_unordered_map<size_t, size_t> the_map;
    concurrent_full_slot_map() {}
    uint32_t register_bucket(size_t slot_array_seq_id) {
      auto rst = the_map.insert({slot_array_seq_id, _place_holder});
      if (rst.second == true) {
        rst.first->second =
            next_free_slot.fetch_add(1); // the allcoated block identifer
      } else {
        while (rst.first->second == _place_holder) {
        }
      }
      return rst.first->second;
    }
  };

  /** ==================================================================
   * @brief status for solving optimal policy based on slot/block slice result
   *  depends on system configuration(i.e. cache size, num device)
   *  ==================================================================
   */

  static inline bool ignore_block(uint32_t block_id, double weight) {
    return (weight == 0) && (block_id > 0);
  }
};

class SingleStreamSolverBase : public CollCacheSolver {
 public:
  void Build(TensorPtr stream_id_list, TensorPtr stream_freq_list,
             std::vector<int> device_to_stream,
             const IdType num_node, const TensorPtr nid_to_block_tensor) override;
  TensorPtr stream_id_list;
  TensorPtr stream_freq_list;
  TensorPtr nid_to_block;
};

class IntuitiveSolver : public SingleStreamSolverBase {
public:
  void Solve(std::vector<int> device_to_stream,
             std::vector<int> device_to_cache_percent, std::string mode,
             double T_local = 2, double T_remote = 15, double T_cpu = 60);
};
class PartitionSolver : public SingleStreamSolverBase {
public:
  void Solve(std::vector<int> device_to_stream,
             std::vector<int> device_to_cache_percent, std::string mode,
             double T_local = 2, double T_remote = 15, double T_cpu = 60);
};
class PartRepSolver : public SingleStreamSolverBase {
public:
  void Solve(std::vector<int> device_to_stream,
             std::vector<int> device_to_cache_percent, std::string mode,
             double T_local = 2, double T_remote = 15, double T_cpu = 60);
};

} // namespace coll_cache
} // namespace common
} // namespace samgraph