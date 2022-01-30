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
#include "common/utils.h"

void CheckCSRValid(utility::GraphPtr graph) {
  size_t num_nodes = graph->num_nodes;
  size_t num_edges = graph->num_edges;
  uint32_t *indptr = graph->indptr;
  uint32_t *indices = graph->indices;

  {
    uint32_t target = 37835909;
    uint32_t off = indptr[target];
    uint32_t len = indptr[target + 1] - off;
    for (uint32_t j = 0; j < len; j++) {
      std::cout << indices[off + j] << std::endl;
    }
  }

#pragma omp parallel for
  for (size_t i = 0; i < num_nodes; i++) {
    uint32_t off = indptr[i];
    uint32_t len = indptr[i + 1] - off;

    utility::Check(len >= 0, "len must be >= 0");
  }

  utility::Check(indptr[num_nodes] == num_edges, "edge number not match");

#pragma omp parallel for
  for (size_t i = 0; i < num_edges; i++) {
    utility::Check(indices[i] >= 0 && indices[i] < num_nodes,
                   "neighbor out of index");
  }
}

void CheckDuplicatedEdges(utility::GraphPtr graph) {
  size_t num_nodes = graph->num_nodes;
  size_t num_edges = graph->num_edges;
  uint32_t *indptr = graph->indptr;
  uint32_t *indices = graph->indices;

  size_t count_duplication = 0;
#pragma omp parallel for
  for (size_t i = 0; i < num_nodes; i++) {
    std::unordered_set<uint32_t> edge_set;
    uint32_t off = indptr[i];
    uint32_t len = indptr[i + 1] - off;

    for (size_t j = 0; j < len; j++) {
      uint32_t neighbor = indices[off + j];
      utility::Check(edge_set.count(neighbor) <= 1, "neighbor out of index");

      edge_set.insert(neighbor);
    }
  }
}

void CheckTrainSet(utility::GraphPtr graph) {
  size_t num_train_set = graph->num_train_set;
  uint32_t *train_set = graph->train_set;

  for (size_t i = 0; i < num_train_set; i++) {
    if (train_set[i] == 4294967295) {
      std::cout << "index " << i << " has node " << train_set[i] << std::endl;
    }
  }
}

int main(int argc, char *argv[]) {
  utility::Options::InitOptions("Graph property");
  OPTIONS_PARSE(argc, argv);

  utility::GraphLoader graph_loader(utility::Options::root);
  auto graph = graph_loader.GetGraphDataset(utility::Options::graph);

  // CheckCSRValid(graph);
  // CheckDuplicatedEdges(graph);
  CheckTrainSet(graph);
}