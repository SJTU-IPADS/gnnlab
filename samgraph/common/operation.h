#ifndef SAMGRAPH_OPERATIONS_H
#define SAMGRAPH_OPERATIONS_H

#include <cstddef>
#include <cstdint>

namespace samgraph {
namespace common {

extern "C" {

void samgraph_init(const char *path, int sample_device_type, int sample_device,
                   int train_device_type, int train_device, size_t batch_size,
                   int *fanout, size_t num_fanout, size_t num_epoch);

void samgraph_start();

void samgraph_shutdown();

size_t samgraph_num_epoch();

size_t samgraph_steps_per_epoch();

size_t samgraph_num_class();

size_t samgraph_feat_dim();

uint64_t samgraph_get_next_batch(uint64_t epoch, uint64_t step);

void samgraph_sample_once();

size_t samgraph_get_graph_num_row(uint64_t key, int graph_id);

size_t samgraph_get_graph_num_col(uint64_t key, int graph_id);

size_t samgraph_get_graph_num_edge(uint64_t key, int graph_id);

void samgraph_profiler_report(uint64_t epoch, uint64_t step);
}

}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_OPERATIONS_H
