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

#include "common/graph_loader.h"
#include "common/options.h"
#include "common/utils.h"

void randkingNodesToFile(utility::GraphPtr graph,
                         std::shared_ptr<utility::DegreeInfo> info) {
  const size_t num_nodes = graph->num_nodes;
  const uint32_t *indptr = graph->indptr;
  const uint32_t *indices = graph->indices;
  const uint32_t *train_set = graph->train_set;
  const size_t num_train_set = graph->num_train_set;
  const std::vector<uint32_t> &out_degrees = info->out_degrees;
  std::vector<std::pair<uint32_t, uint32_t>> outdegree_id_list(num_nodes);

#pragma omp parallel for
  for (uint32_t i = 0; i < num_nodes; i++) {
    outdegree_id_list[i] = {out_degrees[i], i};
  }

#ifdef __linux__
  __gnu_parallel::sort(outdegree_id_list.begin(), outdegree_id_list.end(),
                       std::greater<std::pair<uint32_t, uint32_t>>());
#else
  std::sort(outdegree_id_list.begin(), outdegree_id_list.end(),
            std::greater<std::pair<uint32_t, uint32_t>>());
#endif

  std::vector<uint32_t> ranking_nodes(num_nodes);
  std::vector<bool> added_nodes_bitmap(num_nodes, false);
  uint32_t i = 0;

  // 1. Adding all training nodes first
  for (uint32_t j = 0; j < num_train_set; j++) {
    ranking_nodes[i] = train_set[j];
    added_nodes_bitmap[train_set[j]] = true;
    i++;
  }

  utility::Check(i == num_train_set, "node number mismatch after step 1");

  // 2. Then adding the first-hop neighbors of training nodes
  for (uint32_t j = 0; j < num_train_set; j++) {
    uint32_t node = train_set[j];
    uint32_t off = indptr[node];
    uint32_t len = indptr[node + 1] - indptr[node];

    for (uint32_t k = 0; k < len; k++) {
      uint32_t neighbor = indices[off + k];
      if (!added_nodes_bitmap[neighbor]) {
        ranking_nodes[i] = neighbor;
        added_nodes_bitmap[neighbor] = true;
        i++;
      }
    }
  }

  // 3. Add the rest nodes by out degree
  for (uint32_t j = 0; j < num_nodes && i < num_nodes; j++) {
    uint32_t node = outdegree_id_list[j].second;
    if (!added_nodes_bitmap[node]) {
      ranking_nodes[i] = node;
      added_nodes_bitmap[node] = true;
      i++;
    }
  }

  utility::Check(i == num_nodes, "node number mismatch after step 3");

  std::ofstream ofs(
      graph->folder + "cache_by_heuristic.bin",
      std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);

  ofs.write((const char *)ranking_nodes.data(),
            ranking_nodes.size() * sizeof(uint32_t));

  ofs.close();
}

int main(int argc, char *argv[]) {
  utility::Options::InitOptions("Graph property");
  OPTIONS_PARSE(argc, argv);

  utility::GraphLoader graph_loader(utility::Options::root);
  auto graph = graph_loader.GetGraphDataset(utility::Options::graph);
  auto degree_info = utility::DegreeInfo::GetDegrees(graph);

  randkingNodesToFile(graph, degree_info);
}