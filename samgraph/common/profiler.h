#ifndef SAMGRAPH_PROFILER_H
#define SAMGRAPH_PROFILER_H

#include <vector>

namespace samgraph {
namespace common {

class Profiler {
 public:
  Profiler();

  void Report(size_t epoch, size_t step);
  void ReportAvg(size_t num);

  static size_t GetEntryIndex(size_t epoch, size_t step) {
    return epoch * Get()->_num_step_per_epoch + step;
  }

  static Profiler* Get();

  // Time metric
  std::vector<double> sample_time;
  std::vector<double> ns_time;
  std::vector<double> remap_time;
  std::vector<double> populate_time;
  std::vector<double> map_node_time;
  std::vector<double> map_edge_time;

  // Number metric
  std::vector<size_t> num_samples;

  constexpr static size_t kMaxEntries = 200000;

 private:
  size_t _num_step_per_epoch;
};

}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_PROFILER_H
