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

#include <iostream>
#include <unordered_map>
#include <unordered_set>

#include "common/graph_loader.h"
#include "common/options.h"

void IsDirected(utility::GraphPtr dataset) {
  uint32_t *indptr = dataset->indptr;
  uint32_t *indices = dataset->indices;
  uint32_t num_nodes = dataset->num_nodes;
  uint32_t num_edges = dataset->num_edges;

  std::unordered_map<uint32_t, std::unordered_set<uint32_t>> adj;

  for (uint32_t node_src = 0; node_src < num_nodes; node_src++) {
    adj[node_src] = {};
  }

#pragma omp parallel for
  for (uint32_t node_src = 0; node_src < num_nodes; node_src++) {
    uint32_t start = indptr[node_src];
    uint32_t end = indptr[node_src + 1];

    for (uint32_t j = start; j < end; j++) {
      uint32_t node_dst = indices[j];

      adj[node_src].insert(node_dst);
    }
  }

  size_t count = 0;
#pragma omp parallel for reduction(+ : count)
  for (uint32_t node_src = 0; node_src < num_nodes; node_src++) {
    uint32_t src_start = indptr[node_src];
    uint32_t src_end = indptr[node_src + 1];
    for (uint32_t j = src_start; j < src_end; j++) {
      uint32_t node_dst = indices[j];
      if (adj[node_dst].count(node_src) > 0) {
        count++;
      }
    }
  }

  if (count > 0) {
    std::cout << "The graph is undirected with " << count << " edges"
              << std::endl;
  } else {
    std::cout << "The graph is directed " << std::endl;
  }

  size_t count_duplication = 0;
#pragma omp parallel for reduction(+ : count_duplication)
  for (uint32_t node_src = 0; node_src < num_nodes; node_src++) {
    for (uint32_t node_dst : adj[node_src]) {
      if (adj[node_src].count(node_dst) > 1) {
        count_duplication++;
      }
    }
  }

  std::cout << "The graph has " << count_duplication << " duplicated edges"
            << std::endl;
}

void HasSelfLoop(utility::GraphPtr dataset) {
  uint32_t *indptr = dataset->indptr;
  uint32_t *indices = dataset->indices;
  uint32_t num_nodes = dataset->num_nodes;
  uint32_t num_edges = dataset->num_edges;

  size_t count = 0;
#pragma omp parallel for reduction(+ : count)
  for (uint32_t node_src = 0; node_src < num_nodes; node_src++) {
    uint32_t start = indptr[node_src];
    uint32_t end = indptr[node_src + 1];

    for (uint32_t j = start; j < end; j++) {
      uint32_t node_dst = indices[j];

      if (node_dst == node_src) {
        count++;
      }
    }
  }

  if (count > 0) {
    std::cout << "The graph has " << count << " self-loop " << std::endl;
  } else {
    std::cout << "The graph doesn't has self-loop " << std::endl;
  }
}

void HasZeroDegreeNodes(utility::GraphPtr dataset) {
  uint32_t *indptr = dataset->indptr;
  uint32_t *indices = dataset->indices;
  uint32_t num_nodes = dataset->num_nodes;
  uint32_t num_edges = dataset->num_edges;

  uint32_t count = 0;
  for (uint32_t i = 0; i < num_nodes; i++) {
    uint32_t len = indptr[i + 1] - indptr[i];
    if (len == 0) {
      count++;
    }
  }

  if (count > 0) {
    std::cout << "The graph has " << count << " zero-degree nodes" << std::endl;
  } else {
    std::cout << "The graph doesn't has zero-degree nodes" << std::endl;
  }
}

void IsCSRIndicesSorted(utility::GraphPtr dataset) {
  uint32_t *indptr = dataset->indptr;
  uint32_t *indices = dataset->indices;
  uint32_t num_nodes = dataset->num_nodes;
  uint32_t num_edges = dataset->num_edges;

  uint32_t count = 0;
  for (uint32_t i = 0; i < num_nodes; i++) {
    uint32_t off = indptr[i];
    uint32_t len = indptr[i + 1] - indptr[i];
    uint32_t last_idx = 0;
    for (uint32_t k = 0; k < len; k++) {
      if (indices[off + k] < last_idx) {
        std::cout << "The graph's indices are not sorted" << std::endl;
        return;
      }

      last_idx = indices[off + k];
    }
  }

  std::cout << "The graph's indices are sorted" << std::endl;
}

int main(int argc, char *argv[]) {
  utility::Options::InitOptions("Graph property");
  OPTIONS_PARSE(argc, argv);

  utility::GraphLoader graph_loader(utility::Options::root);
  auto graph = graph_loader.GetGraphDataset(utility::Options::graph);

  IsDirected(graph);
  HasSelfLoop(graph);
  HasZeroDegreeNodes(graph);
  IsCSRIndicesSorted(graph);

  return 0;
}