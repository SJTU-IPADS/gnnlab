#ifndef SAMGRAPH_OPERATION_H
#define SAMGRAPH_OPERATION_H

#include <cstddef>
#include <cstdint>

namespace samgraph {
namespace common {

extern "C" {

void samgraph_config(const char *path, int run_arch, int sample_type,
                     int sampler_device_type, int sampler_device_id,
                     int trainer_device_type, int trainer_device_id,
                     size_t batch_size, size_t num_epoch, int cache_policy,
                     double cache_percentage, size_t max_sampling_jobs,
                     size_t max_copying_jobs);

void samgraph_config_khop(size_t *fanout, size_t num_fanout);

void samgraph_config_random_walk(size_t random_walk_length,
                                 double random_walk_restart_prob,
                                 size_t num_random_walk, size_t num_neighbor,
                                 size_t num_layer);

void samgraph_init();

void samgraph_start();

void samgraph_shutdown();

size_t samgraph_num_epoch();

size_t samgraph_steps_per_epoch();

size_t samgraph_num_class();

size_t samgraph_feat_dim();

uint64_t samgraph_get_next_batch();

void samgraph_sample_once();

size_t samgraph_get_graph_num_src(uint64_t key, int graph_id);

size_t samgraph_get_graph_num_dst(uint64_t key, int graph_id);

size_t samgraph_get_graph_num_edge(uint64_t key, int graph_id);

void samgraph_log_step(uint64_t epoch, uint64_t step, int item, double val);

void samgraph_log_step_add(uint64_t epoch, uint64_t step, int item, double val);

void samgraph_log_epoch_add(uint64_t epoch, int item, double val);

double samgraph_get_log_step_value(uint64_t epoch, uint64_t step, int item);

double samgraph_get_log_epoch_value(uint64_t epoch, int item);

void samgraph_report_init();

void samgraph_report_step(uint64_t epoch, uint64_t step);

void samgraph_report_step_average(uint64_t epoch, uint64_t step);

void samgraph_report_epoch(uint64_t epoch);

void samgraph_report_epoch_average(uint64_t epoch);

void samgraph_report_node_access();

// for multi-GPUs train, call data_init before fork
void samgraph_data_init();
void samgraph_sample_init(int device_type, int device_id);
void samgraph_train_init(int device_type, int device_id);
void samgraph_sample();
void samgraph_extract();
void samgraph_extract_start(int count);


void samgraph_trace_step_begin(uint64_t key, int item, uint64_t ts);

void samgraph_trace_step_end(uint64_t key, int item, uint64_t ts);

void samgraph_trace_step_begin_now(uint64_t key, int item);

void samgraph_trace_step_end_now(uint64_t key, int item);

void samgraph_dump_trace();

void samgraph_forward_barrier();
}

}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_OPERATION_H
