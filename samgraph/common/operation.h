#ifndef SAMGRAPH_OPERATIONS_H
#define SAMGRAPH_OPERATIONS_H

namespace samgraph {
namespace common {

extern "C" {

void samgraph_init(const char*path, int sample_device, int train_device,
                   size_t batch_size, int *fanout, size_t num_fanout, int num_epoch);

void samgraph_start();

void samgraph_shutdown();

int samgraph_num_epoch();

int samgraph_num_step_per_epoch();

size_t samgraph_dataset_num_class();

size_t samgraph_dataset_num_feat_dim();

uint64_t samgraph_get_next_batch(int epoch, int step);

uint64_t samgraph_get_graph_key(uint64_t batch_key, int graph_id);

size_t samgraph_get_graph_num_row(uint64_t key);

size_t samgraph_get_graph_num_col(uint64_t key);

size_t samgraph_get_graph_num_edge(uint64_t key);

}

} // namespace common
} // namespace samgraph

#endif // SAMGRAPH_OPERATIONS_H
