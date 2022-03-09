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

#include "common/graph_loader.h"
#include "common/options.h"
#include "common/utils.h"


volatile uint8_t* hopNodes(utility::GraphPtr dataset, size_t hop) {
  uint32_t *indptr = dataset->indptr;
  uint32_t *indices = dataset->indices;
  uint32_t *train_set = dataset->train_set;
  uint32_t num_train_set = dataset->num_train_set;
  uint32_t num_nodes = dataset->num_nodes;
  uint32_t num_edges = dataset->num_edges;

  // std::vector<std::uint8_t> before(num_nodes, 0);
  volatile uint8_t*  before = new volatile uint8_t [num_nodes]();
  volatile uint8_t*  after = new volatile uint8_t [num_nodes]();

#pragma omp parallel for
  for (uint32_t cur_node = 0; cur_node < num_nodes; cur_node++) {
    before[cur_node] = 0;
    after[cur_node] = 0;
  }
#pragma omp parallel for
  for (uint32_t idx = 0; idx < num_train_set; idx++) {
    uint32_t node_hop0 = train_set[idx];
    before[node_hop0] = 2;
  }
  for (size_t hop_idx = 0; hop_idx < hop ; hop_idx++) {
#pragma omp parallel for
    for (uint32_t cur_node = 0; cur_node < num_nodes; cur_node++) {
      if (before[cur_node] != 0x02) continue;

      uint32_t start = indptr[cur_node];
      uint32_t end = indptr[cur_node + 1];

      for (uint32_t j = start; j < end; j++) {
        uint32_t node_dst = indices[j];
        after[node_dst] = 1;
      }
    }
#pragma omp parallel for
    for (uint32_t cur_node = 0; cur_node < num_nodes; cur_node++) {
      if (after[cur_node] == 0) {
        if (before[cur_node]) before[cur_node] = 1;
      } else {
        if (before[cur_node]) { // 1 or 2
          before[cur_node] = 1;
        } else {
          before[cur_node] = 2;
        }
        after[cur_node] = 0;
      }
    }
  }
  return before;
}


utility::GraphPtr gen_khop_graph(utility::GraphPtr orig_graph, uint8_t* required_nodes) {
  size_t num_nodes = orig_graph->num_nodes;
  utility::GraphPtr new_graph = std::make_shared<utility::Graph>(*orig_graph);

  uint32_t* new_indptr = new uint32_t[num_nodes+1];
  new_indptr[0] = 0;
#pragma omp parallel for
  for (uint32_t cur_node = 0; cur_node < num_nodes; cur_node++) {
    if (required_nodes[cur_node] == 0) continue;
    size_t deg = orig_graph->indptr[cur_node + 1] - orig_graph->indptr[cur_node];
    new_indptr[cur_node + 1] = deg;
  }
  for (int i = 1; i < num_nodes + 1; i++) {
    new_indptr[i] += new_indptr[i-1];
  }
  size_t new_num_edge = new_indptr[num_nodes];
  uint32_t* new_indices = new uint32_t[new_num_edge];

#pragma omp parallel for
  for (uint32_t cur_node = 0; cur_node < num_nodes; cur_node++) {
    if (required_nodes[cur_node] == 0) continue;
    
    uint32_t old_start = orig_graph->indptr[cur_node];
    uint32_t old_end = orig_graph->indptr[cur_node + 1];
    uint32_t new_j = new_indptr[cur_node];
    for (uint32_t j = old_start; j < old_end; j++) {
      uint32_t node_dst = orig_graph->indices[j];
      new_indices[new_j] = node_dst;
      new_j++;
    }
  }
  new_graph->indptr = new_indptr;
  new_graph->indices = new_indices;
  return new_graph;
}

void merge_degree_info(
    std::shared_ptr<utility::DegreeInfo> whole_g, 
    std::shared_ptr<utility::DegreeInfo> sub_g, 
    uint8_t* touched_node) {
#pragma omp parallel for
  for (uint32_t cur_node = 0; cur_node < sub_g->in_degrees.size(); cur_node++) {
    if (touched_node[cur_node] == 0) continue;
    whole_g->in_degrees[cur_node] = sub_g->in_degrees[cur_node] | 0x40000000;
    whole_g->out_degrees[cur_node] = sub_g->out_degrees[cur_node] | 0x40000000;
  }
}

void randkingNodesToFile(utility::GraphPtr graph,
                         std::shared_ptr<utility::DegreeInfo> info) {
  size_t num_nodes = graph->num_nodes;
  const std::vector<uint32_t>& out_degrees = info->out_degrees;
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

#pragma omp parallel for
  for (size_t i = 0; i < num_nodes; i++) {
    ranking_nodes[i] = outdegree_id_list[i].second;
  }

  std::ofstream ofs(
      graph->folder + "cache_by_degree_hop.bin",
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
  auto orig_degree_info = utility::DegreeInfo::GetDegrees(orig_graph);

  auto touched_nodes = const_cast<uint8_t*>(hopNodes(orig_graph, 2));
  auto new_graph = gen_khop_graph(orig_graph, touched_nodes);
  auto new_degree_info = utility::DegreeInfo::GetDegrees(new_graph);
  merge_degree_info(orig_degree_info, new_degree_info, touched_nodes);

  randkingNodesToFile(orig_graph, orig_degree_info);
}