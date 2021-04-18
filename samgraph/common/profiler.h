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

  // Time breakdown level 0
  std::vector<double> sample_time;
  std::vector<double> extract_time;
  std::vector<double> copy_time;
  std::vector<double> train_time;

  // Time breakdown level 1
  std::vector<double> ns_time;
  std::vector<double> remap_time;

  // Time breakdown level 2
  std::vector<double> graph_copy_time;
  std::vector<double> feat_copy_time;

  // Time breakdown level 2
  std::vector<double> sample_calculation_time;
  std::vector<double> sample_count_edge_time;
  std::vector<double> sample_compact_edge_time;

  std::vector<double> populate_time;
  std::vector<double> map_node_time;
  std::vector<double> map_edge_time;
  std::vector<double> coo2csr_time;

  // Time breakdown
  //   std::vector<double>;

  std::vector<double> alloc_val_time;
  std::vector<double> fill_val_time;
  std::vector<double> csrmm_time;
  std::vector<double> csrmm_transpose_time;

  // Number of samples
  std::vector<size_t> num_samples;

  constexpr static size_t kMaxEntries = 200000;

 private:
  size_t _num_step_per_epoch;
};

}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_PROFILER_H
