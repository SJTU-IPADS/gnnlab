#ifndef SAMGRAPH_PROFILER_H
#define SAMGRAPH_PROFILER_H

#include <cstdint>
#include <vector>

namespace samgraph {
namespace common {

class Profiler {
 public:
  Profiler();

  void Report(uint64_t key);
  void ReportAverage(size_t num);

  static Profiler* Get();

  // Top Level
  std::vector<size_t> num_samples;
  std::vector<double> sample_time;
  std::vector<double> copy_time;

  // Second Level
  std::vector<double> shuffle_time;
  std::vector<double> real_sample_time;

  std::vector<double> graph_copy_time;
  std::vector<double> id_copy_time;
  std::vector<double> extract_time;
  std::vector<double> feat_copy_time;

  std::vector<double> ns_time;
  std::vector<double> remap_time;

  std::vector<double> sample_calculation_time;
  std::vector<double> sample_count_edge_time;
  std::vector<double> sample_compact_edge_time;

  std::vector<double> populate_time;
  std::vector<double> map_node_time;
  std::vector<double> map_edge_time;

 private:
  size_t _max_entries;
};

}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_PROFILER_H
