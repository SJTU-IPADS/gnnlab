#include "common/common.h"
#include "common/run_config.h"
#include <vector>

namespace samgraph {
namespace common {
namespace coll_cache {

void solve_rep(TensorPtr stream_id_list, TensorPtr stream_freq_list, const IdType num_node,
    std::vector<int> device_to_stream, std::vector<int> device_to_cache_percent,
    TensorPtr nid_to_block, TensorPtr & block_placement,
    std::string _, double T_local = 2, double T_remote = 15, double T_cpu = 60);
void solve_selfish(TensorPtr stream_id_list, TensorPtr stream_freq_list, const IdType num_node,
    std::vector<int> device_to_stream, std::vector<int> device_to_cache_percent,
    TensorPtr nid_to_block, TensorPtr & block_placement,
    std::string _, double T_local = 2, double T_remote = 15, double T_cpu = 60);
void solve_partition_rep_diff_freq(TensorPtr stream_id_list, TensorPtr stream_freq_list, const IdType num_node,
    std::vector<int> device_to_stream, std::vector<int> device_to_cache_percent,
    TensorPtr nid_to_block, TensorPtr & block_placement,
    std::string _, double T_local = 2, double T_remote = 15, double T_cpu = 60);
}
}
}