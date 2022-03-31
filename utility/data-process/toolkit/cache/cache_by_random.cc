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
#include <random>

#include "common/graph_loader.h"
#include "common/options.h"
#include "common/utils.h"

void randkingNodesToFile(utility::GraphPtr graph) {
  size_t num_nodes = graph->num_nodes;
  std::vector<uint32_t> ranking_nodes(num_nodes);

#pragma omp parallel for
  for (uint32_t i = 0; i < num_nodes; i++) {
    ranking_nodes[i] = {i};
  }

  std::mt19937 generator;
  for (uint32_t i = 0; i < num_nodes; i++) {
    std::uniform_int_distribution<uint32_t> distribution(0, num_nodes - i - 1);
    std::swap(ranking_nodes[num_nodes - i - 1], ranking_nodes[distribution(generator)]);
  }

  std::ofstream ofs(
      graph->folder + "cache_by_random.bin",
      std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);

  ofs.write((const char*)ranking_nodes.data(),
            ranking_nodes.size() * sizeof(uint32_t));

  ofs.close();
}

int main(int argc, char* argv[]) {
  utility::Options::InitOptions("Graph property");
  OPTIONS_PARSE(argc, argv);

  utility::GraphLoader graph_loader(utility::Options::root);
  auto graph = graph_loader.GetGraphDataset(utility::Options::graph);

  randkingNodesToFile(graph);
}