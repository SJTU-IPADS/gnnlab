#include "profiler.h"

#include <cstdio>

#include "engine.h"
#include "macros.h"

namespace samgraph {
namespace common {

Profiler::Profiler() {
  auto num_epoch = Engine::Get()->NumEpoch();
  auto num_step_per_epoch = Engine::Get()->NumStep();

  _max_entries = num_epoch * num_step_per_epoch;

  num_samples.resize(_max_entries, 0);
  sample_time.resize(_max_entries, 0);
  copy_time.resize(_max_entries, 0);

  shuffle_time.resize(_max_entries, 0);
  real_sample_time.resize(_max_entries, 0);

  graph_copy_time.resize(_max_entries, 0);
  id_copy_time.resize(_max_entries, 0);
  extract_time.resize(_max_entries, 0);
  feat_copy_time.resize(_max_entries, 0);

  ns_time.resize(_max_entries, 0);
  remap_time.resize(_max_entries, 0);

  graph_copy_time.resize(_max_entries, 0);
  feat_copy_time.resize(_max_entries, 0);

  sample_calculation_time.resize(_max_entries, 0);
  sample_count_edge_time.resize(_max_entries, 0);
  sample_compact_edge_time.resize(_max_entries, 0);

  populate_time.resize(_max_entries, 0);
  map_node_time.resize(_max_entries, 0);
  map_edge_time.resize(_max_entries, 0);
}

void Profiler::Report(uint64_t key) {
  uint64_t epoch = Engine::Get()->GetEpochFromKey(key);
  uint64_t step = Engine::Get()->GetStepFromKey(key);

#if PRINT_PROFILE
  printf(
      "  [Profile] Epoch %llu | step %llu | num samples: %lu | sample time: "
      "%.4lf | copy time: %.4lf \n",
      epoch, step, num_samples[key], sample_time[key], copy_time[key]);

  printf("    [Sample Profile] | shuffle %.4lf | real sampling %.4lf \n",
         shuffle_time[key], real_sample_time[key]);

  printf(
      "    [Copy Profile] | graph copy %.4lf  | id copy %.4lf  | extract %.4lf "
      " | feat copy %.4lf \n",
      graph_copy_time[key], id_copy_time[key], extract_time[key],
      feat_copy_time[key]);
#endif
}

void Profiler::ReportAverage(size_t num) {}

Profiler* Profiler::Get() {
  static Profiler inst;
  return &inst;
}

}  // namespace common
}  // namespace samgraph
