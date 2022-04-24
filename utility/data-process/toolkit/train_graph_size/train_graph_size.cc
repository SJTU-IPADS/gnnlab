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

/**
 * calculate the graph size of using only one batch
 */

#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <cassert>
#include <limits>
#include <random>
#include <cstring>

#include "common/graph_loader.h"
#include "common/options.h"
#include "common/utils.h"

uint32_t max_hop = std::numeric_limits<uint32_t>::max();


void shuffle(uint32_t * data, size_t num_data, uint64_t seed= 0x1234567890abcdef) {
  auto g = std::default_random_engine(seed);
  for (size_t i = num_data - 1; i > 0; i--) {
    std::uniform_int_distribution<size_t> d(0, i);
    size_t candidate = d(g);
    std::swap(data[i], data[candidate]);
  }
}

uint32_t* RandSet(uint32_t num_nodes, uint64_t seed = 0x1234567890abcdef) {
  uint32_t *id_list = new uint32_t[num_nodes];
#pragma omp parallel for
  for (uint32_t cur_node = 0; cur_node < num_nodes; cur_node++) {
    id_list[cur_node] = cur_node;
  }

  shuffle(id_list, num_nodes);
  return id_list;
}

void TrainSize(utility::GraphPtr dataset, int partition, double percent) {
  uint32_t *indptr = dataset->indptr;
  uint32_t *indices = dataset->indices;
  uint32_t *train_set = dataset->train_set;
  uint32_t num_train_set = dataset->num_train_set;
  uint32_t num_nodes = dataset->num_nodes;
  uint32_t num_edges = dataset->num_edges;
  std::cout << "Graph has " << num_train_set << " train set\n";

  if (!std::isnan(percent)) {
    num_train_set = ((uint64_t)num_nodes) * percent / 100;
    if (num_train_set > dataset->num_train_set) {
      train_set = RandSet(num_nodes);
    } else {
      uint32_t *old_train_set = train_set;
      train_set = new uint32_t[dataset->num_train_set];
      std::memcpy(train_set, old_train_set, dataset->num_train_set * sizeof(uint32_t));
      shuffle(train_set, dataset->num_train_set);
    }
  }

  num_train_set /= partition;

  // std::vector<std::uint8_t> before(num_nodes, 0);
  volatile uint8_t*  before = new volatile uint8_t [num_nodes]();
  volatile uint8_t*  after = new volatile uint8_t [num_nodes]();

  std::printf("On hop 0: count %12d/%d (%4.2lf%%) nodes\n", num_train_set, num_nodes, num_train_set * ((double)100) / num_nodes);
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

  // const size_t hop = 6;
  size_t last_count = 0;
  for (size_t hop_idx = 0; hop_idx < max_hop; hop_idx++) {
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

    size_t count = 0;
    for (uint32_t node_src = 0; node_src < num_nodes; node_src++) {
      if (before[node_src] != 0) count++;
    }
    std::printf("On hop %lu: count %12lu/%d (%4.2lf%%) nodes\n", hop_idx+1, count, num_nodes, (count * (double)100) / num_nodes);
    if (last_count == count) break;
    else {
      last_count = count;
    }
  }
}

int main(int argc, char *argv[]) {
  utility::Options::InitOptions("Graph property");
  int partition = 1;
  double percent = std::nan("");
  utility::Options::CustomOption("--part", partition, "partition train set. ignored when <percent> is set.");
  utility::Options::CustomOption("--percent", percent, "use <percent>%% of entire graph as train set.");
  utility::Options::CustomOption("--max-hop", max_hop, "max hop to check.");
  OPTIONS_PARSE(argc, argv);

  utility::GraphLoader graph_loader(utility::Options::root);
  auto graph = graph_loader.GetGraphDataset(utility::Options::graph);

  TrainSize(graph, partition, percent);

  return 0;
}