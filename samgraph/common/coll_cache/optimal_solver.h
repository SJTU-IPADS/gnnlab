#include "../common.h"
#include "../run_config.h"
#include "../logging.h"
#include <vector>

namespace samgraph {
namespace common {
namespace coll_cache {

inline size_t num_blocks(int num_stream, int num_slot) {
  auto shape = std::vector<size_t>(num_stream, num_slot);
  return std::accumulate(shape.begin(), shape.end(), 1ul, std::multiplies<size_t>());
}
void solve(TensorPtr stream_id_list, TensorPtr stream_freq_list, const IdType num_node,
    std::vector<int> device_to_stream, std::vector<int> device_to_cache_percent,
    TensorPtr nid_to_block, TensorPtr & block_placement,
    std::string mode, double T_local = 2, double T_remote = 15, double T_cpu = 60);
void solve_intuitive(TensorPtr stream_id_list, TensorPtr stream_freq_list, const IdType num_node,
    std::vector<int> device_to_stream, std::vector<int> device_to_cache_percent,
    TensorPtr nid_to_block, TensorPtr & block_placement,
    std::string mode, double T_local = 2, double T_remote = 15, double T_cpu = 60);
void solve_partition(TensorPtr stream_id_list, TensorPtr stream_freq_list, const IdType num_node,
    std::vector<int> device_to_stream, std::vector<int> device_to_cache_percent,
    TensorPtr nid_to_block, TensorPtr & block_placement,
    std::string _, double T_local = 2, double T_remote = 15, double T_cpu = 60);
void solve_partition_rep(TensorPtr stream_id_list, TensorPtr stream_freq_list, const IdType num_node,
    std::vector<int> device_to_stream, std::vector<int> device_to_cache_percent,
    TensorPtr nid_to_block, TensorPtr & block_placement,
    std::string _, double T_local = 2, double T_remote = 15, double T_cpu = 60);
}
}
}