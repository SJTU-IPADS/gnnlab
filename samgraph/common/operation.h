/*
 * Copyright 2022 Institute of Parallel and Distributed Systems, Shanghai Jiao Tong University
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#ifndef SAMGRAPH_OPERATION_H
#define SAMGRAPH_OPERATION_H

#include <cstddef>
#include <cstdint>
#include <unordered_map>

namespace samgraph {
namespace common {

extern "C" {

void samgraph_config(const char **config_keys, const char **config_values,
                     const size_t num_config_items);

void samgraph_config_from_map(std::unordered_map<std::string, std::string>& configs);

void samgraph_init();

void samgraph_start();

void samgraph_shutdown();

size_t samgraph_num_epoch();

size_t samgraph_steps_per_epoch();

size_t samgraph_num_class();

size_t samgraph_feat_dim();

uint64_t samgraph_get_next_batch();

void samgraph_unset_cur_batch();

void samgraph_sample_once();

size_t samgraph_get_graph_num_src(uint64_t key, int graph_id);

size_t samgraph_get_graph_num_dst(uint64_t key, int graph_id);

size_t samgraph_get_graph_num_edge(uint64_t key, int graph_id);

size_t samgraph_get_unsupervised_graph_num_node(uint64_t key);

void samgraph_log_step(uint64_t epoch, uint64_t step, int item, double val);
void samgraph_log_step_by_key(uint64_t key, int item, double val);

void samgraph_log_step_add(uint64_t epoch, uint64_t step, int item, double val);

void samgraph_log_epoch_add(uint64_t epoch, int item, double val);

double samgraph_get_log_step_value(uint64_t epoch, uint64_t step, int item);
double samgraph_get_log_step_value_by_key(uint64_t key, int item);

double samgraph_get_log_epoch_value(uint64_t epoch, int item);

void samgraph_report_init();

void samgraph_report_step(uint64_t epoch, uint64_t step);

void samgraph_report_step_average(uint64_t epoch, uint64_t step);
void samgraph_report_step_max(uint64_t epoch, uint64_t step);
void samgraph_report_step_min(uint64_t epoch, uint64_t step);

void samgraph_report_epoch(uint64_t epoch);

void samgraph_report_epoch_average(uint64_t epoch);

void samgraph_report_node_access();

void samgraph_trace_step_begin(uint64_t key, int item, uint64_t ts);

void samgraph_trace_step_end(uint64_t key, int item, uint64_t ts);

void samgraph_trace_step_begin_now(uint64_t key, int item);

void samgraph_trace_step_end_now(uint64_t key, int item);

void samgraph_dump_trace();

void samgraph_forward_barrier();

// for multi-GPUs train, call data_init before fork
void samgraph_data_init();

void samgraph_sample_init(int worker_id, const char*ctx);

void samgraph_um_sample_init(int num_workers);

void samgraph_train_init(int worker_id, const char*ctx);

void samgraph_sample();

void samgraph_extract();

void samgraph_extract_start(int count);

// for dynamic switch
void samgraph_switch_init(int worker_id, const char*ctx, double cache_percentage);

void samgraph_train_barrier();

size_t samgraph_num_local_step();

int samgraph_wait_one_child();

void samgraph_reset_progress();

void samgraph_print_memory_usage();
}

}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_OPERATION_H
