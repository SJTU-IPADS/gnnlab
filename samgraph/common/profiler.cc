#include "profiler.h"

#include <cstdio>

#include "engine.h"

namespace samgraph {
namespace common {

Profiler::Profiler() {
  sample_time.resize(kMaxEntries, 0);
  ns_time.resize(kMaxEntries, 0);
  remap_time.resize(kMaxEntries, 0);
  populate_time.resize(kMaxEntries, 0);
  map_node_time.resize(kMaxEntries, 0);
  map_edge_time.resize(kMaxEntries, 0);
  num_samples.resize(kMaxEntries, 0);

  _num_step_per_epoch = SamGraphEngine::GetEngine()->GetNumStep();
}

void Profiler::Report(size_t epoch, size_t step) {
  size_t idx = GetEntryIndex(epoch, step);
  printf(
      "  [PROFILE] Epoch %lu | step %lu | total: %.4lf | num samples: %lu | "
      "ns: %.4lf | remap: %.4lf | populate %.4lf | map nodes: %.4lf | map "
      "edges %.4lf |\n",
      epoch, step, sample_time[idx], num_samples[idx], ns_time[idx],
      remap_time[idx], populate_time[idx], map_node_time[idx],
      map_edge_time[idx]);
}

void Profiler::ReportAvg(size_t num) {}

Profiler* Profiler::Get() {
  static Profiler inst;
  return &inst;
}

}  // namespace common
}  // namespace samgraph
