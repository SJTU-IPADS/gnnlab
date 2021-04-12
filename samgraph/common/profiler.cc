#include "profiler.h"

#include <cstdio>

#include "engine.h"
#include "macros.h"

namespace samgraph {
namespace common {

Profiler::Profiler() {
  sample_time.resize(kMaxEntries, 0);
  extract_time.resize(kMaxEntries, 0);
  copy_time.resize(kMaxEntries, 0);
  train_time.resize(kMaxEntries, 0);

  ns_time.resize(kMaxEntries, 0);
  remap_time.resize(kMaxEntries, 0);
  coo2csr_time.resize(kMaxEntries, 0);

  graph_copy_time.resize(kMaxEntries, 0);
  feat_copy_time.resize(kMaxEntries, 0);

  sample_calculation_time.resize(kMaxEntries, 0);
  sample_count_edge_time.resize(kMaxEntries, 0);
  sample_compact_edge_time.resize(kMaxEntries, 0);

  populate_time.resize(kMaxEntries, 0);
  map_node_time.resize(kMaxEntries, 0);
  map_edge_time.resize(kMaxEntries, 0);

  num_samples.resize(kMaxEntries, 0);

  _num_step_per_epoch = SamGraphEngine::GetEngine()->GetNumStep();
}

void Profiler::Report(size_t epoch, size_t step) {
  size_t idx = GetEntryIndex(epoch, step);
#if PRINT_PROFILE
  printf(
      "  [PROFILE] Epoch %lu | step %lu | num samples: %lu | sample time: "
      "%.4lf | extract time: %.4lf | copy time: %.4lf \n",
      epoch, step, num_samples[idx], sample_time[idx], extract_time[idx],
      copy_time[idx]);
  printf(
      "    [SAMPLE PROFILE] | ns time: %.4lf | remap time: %.4lf | coo2csr "
      "time %.4lf \n",
      ns_time[idx], remap_time[idx], coo2csr_time[idx]);
#endif
  // "ns: %.4lf | remap: %.4lf | populate %.4lf | map nodes: %.4lf | map "
  // "edges %.4lf |\n",
  // printf(
  //     "    [SAMPLE] total %.4lf | sample %.4lf | count edge %.4lf | compact "
  //     "edge %.4lf \n",
  //     total, sample_time, count_edge_time, compact_edge_time);
  // printf("");
}

void Profiler::ReportAvg(size_t num) {}

Profiler* Profiler::Get() {
  static Profiler inst;
  return &inst;
}

}  // namespace common
}  // namespace samgraph
