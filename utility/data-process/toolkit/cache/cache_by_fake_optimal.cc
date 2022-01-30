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

#include <fstream>
#ifdef __linux__
#include <parallel/algorithm>
#else
#include <algorithm>
#endif
#include <cassert>
#include <atomic>
#include <vector>

#include "common/graph_loader.h"
#include "common/options.h"
#include "common/utils.h"

class TouchedNodeCtx {
 public:
  std::vector<std::vector<uint32_t>> touched_nodes;
  std::vector<std::vector<bool>> node_touched;
  TouchedNodeCtx(size_t num_nodes) : 
      touched_nodes(utility::Options::num_threads, std::vector<uint32_t>()),
      node_touched(utility::Options::num_threads, std::vector<bool>(num_nodes, false)) {
  }
  TouchedNodeCtx(size_t num_nodes, size_t num_threads) : 
      touched_nodes(num_threads, std::vector<uint32_t>()),
      node_touched(num_threads, std::vector<bool>(num_nodes, false)) {
  }
  void touch(uint32_t node_id) {
    touch(node_id, node_id % utility::Options::num_threads);
  }
  void touch(uint32_t node_id, uint32_t thread_idx) {
    if (node_touched[thread_idx][node_id]) return;
    node_touched[thread_idx][node_id] = true;
    touched_nodes[thread_idx].push_back(node_id);
  }
  std::vector<uint32_t> compact() {
    std::vector<uint32_t> ret;
    for (auto & tn : touched_nodes) {
      for (uint32_t node_id : tn) {
        ret.push_back(node_id);
      }
    }
    return ret;
  }
};

void procBatchTrainNode(double * expection_table, utility::GraphPtr graph, uint32_t train_idx_begin, uint32_t train_idx_end, std::vector<int> fanout) {
  // 1st hop
  auto touch_ctx = TouchedNodeCtx(graph->num_nodes);
  for (uint32_t train_idx = train_idx_begin; train_idx < train_idx_end; train_idx++) {
    uint32_t train_node = graph->train_set[train_idx];
    touch_ctx.touch(train_node);
  }
  std::vector<double> hop1_miss_prob_table(graph->num_nodes, 1);
  {
    std::chrono::time_point<std::chrono::steady_clock> _start_time = std::chrono::steady_clock::now();
#pragma omp parallel for
    for (int thread_idx = 0; thread_idx < utility::Options::num_threads; thread_idx++) {
      for (uint32_t train_idx = train_idx_begin; train_idx < train_idx_end; train_idx++) {
        uint32_t train_node = graph->train_set[train_idx];
        uint32_t deg = graph->indptr[train_node + 1] - graph->indptr[train_node];
        double miss_prob = 1 - fanout[1] / static_cast<double>(deg);
        miss_prob = std::max(0.0, miss_prob);
        for (uint32_t j = graph->indptr[train_node]; j < graph->indptr[train_node+1]; j++) {
          uint32_t dst_node = graph->indices[j];
          if (dst_node % utility::Options::num_threads != thread_idx) continue;
          hop1_miss_prob_table[dst_node] *= miss_prob;
          touch_ctx.touch(dst_node);
        }
      }
    }
#pragma omp parallel for
    for (uint32_t train_idx = train_idx_begin; train_idx < train_idx_end; train_idx++) {
      uint32_t train_node = graph->train_set[train_idx];
      hop1_miss_prob_table[train_node] = 0.0;
    }
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - _start_time).count() << "\n";
  }
  std::vector<double> hop2_miss_prob_table(graph->num_nodes, 1);
  {
    std::chrono::time_point<std::chrono::steady_clock> _start_time = std::chrono::steady_clock::now();
    std::vector<uint32_t> touched_nodes = touch_ctx.compact();
#pragma omp parallel for
    for (int thread_idx = 0; thread_idx < utility::Options::num_threads; thread_idx++) {
      for (uint32_t j = 0; j < touched_nodes.size(); j++) {
        uint32_t hop1_node = touched_nodes[j];
        uint32_t deg = graph->indptr[hop1_node + 1] - graph->indptr[hop1_node];
        double b1_hit = 1 - hop1_miss_prob_table[hop1_node];
        double b2_hit = std::min(1.0, fanout[0] / static_cast<double>(deg));
        double path_miss = 1 - b1_hit * b2_hit;
        for (uint32_t k = graph->indptr[hop1_node]; k < graph->indptr[hop1_node+1]; k++) {
          uint32_t hop2_node = graph->indices[k];
          if (hop2_node % utility::Options::num_threads != thread_idx) continue;
          hop2_miss_prob_table[hop2_node] *= path_miss;
          touch_ctx.touch(hop2_node);
        }
      }
    }
#pragma omp parallel for
    for (uint32_t train_idx = train_idx_begin; train_idx < train_idx_end; train_idx++) {
      uint32_t train_node = graph->train_set[train_idx];
      hop2_miss_prob_table[train_node] = 0.0;
    }
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - _start_time).count() << "\n";
  }
  std::chrono::time_point<std::chrono::steady_clock> _start_time = std::chrono::steady_clock::now();
  std::vector<uint32_t> touched_nodes = touch_ctx.compact();
#pragma omp parallel for
  for (size_t idx = 0; idx < touched_nodes.size(); idx++) {
    uint32_t cur_node = touched_nodes[idx];
    if (hop1_miss_prob_table[cur_node] == 1 && hop2_miss_prob_table[cur_node] == 1) continue;
    expection_table[cur_node] += 1 - hop1_miss_prob_table[cur_node] * hop2_miss_prob_table[cur_node];
  }
  std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - _start_time).count() << "\n";
}

void randkingNodesToFile(utility::GraphPtr graph,
                        double* info) {
  size_t num_nodes = graph->num_nodes;
  std::vector<std::pair<double, uint32_t>> info_id_list(num_nodes);

#pragma omp parallel for
  for (uint32_t i = 0; i < num_nodes; i++) {
    info_id_list[i] = {info[i], i};
  }

#ifdef __linux__
  __gnu_parallel::sort(info_id_list.begin(), info_id_list.end(),
                       std::greater<std::pair<double, uint32_t>>());
#else
  std::sort(info_id_list.begin(), info_id_list.end(),
            std::greater<std::pair<uint32_t, uint32_t>>());
#endif

  std::vector<uint32_t> ranking_nodes(num_nodes);

#pragma omp parallel for
  for (size_t i = 0; i < num_nodes; i++) {
    ranking_nodes[i] = info_id_list[i].second;
  }

  std::ofstream ofs(
      graph->folder + "cache_by_fake_optimal.bin",
      std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);

  ofs.write((const char*)ranking_nodes.data(),
            ranking_nodes.size() * sizeof(uint32_t));

  ofs.close();
}

int main(int argc, char* argv[]) {
  utility::Options::InitOptions("Graph property");
  OPTIONS_PARSE(argc, argv);

  utility::GraphLoader graph_loader(utility::Options::root);
  auto orig_graph = graph_loader.GetGraphDataset(utility::Options::graph);
  std::vector<int> fanout = {25, 10};

  std::vector<double> expection_table(orig_graph->num_nodes, 0);
  // size_t batch_size = 8000;
  size_t batch_size = 1;
  for (size_t i = 0; i < orig_graph->num_train_set; i += batch_size) {
    procBatchTrainNode(expection_table.data(), orig_graph, i, std::min(i + batch_size, orig_graph->num_train_set), fanout);
    std::cout << "done " << i << "/" << orig_graph->num_train_set << "\n";
  }

  randkingNodesToFile(orig_graph, expection_table.data());
}