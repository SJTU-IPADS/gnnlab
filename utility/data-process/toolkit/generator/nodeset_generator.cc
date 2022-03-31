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
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/graph_loader.h"
#include "common/options.h"
#include "common/utils.h"

void generateNodeSet(utility::GraphPtr graph) {
  size_t num_nodes = graph->num_nodes;
  size_t num_edges = graph->num_edges;
  size_t num_train_set = graph->num_train_set;
  size_t num_valid_set = graph->num_valid_set;
  size_t num_test_set = graph->num_test_set;

  std::string output_dir = graph->folder;

  uint32_t *indptr = graph->indptr;
  uint32_t *indices = graph->indices;

  std::vector<bool> bitmap(num_nodes, false);
  std::vector<uint32_t> train_set;
  std::vector<uint32_t> test_set;
  std::vector<uint32_t> valid_set;

  train_set.reserve(num_train_set);
  test_set.reserve(num_test_set);
  valid_set.reserve(num_valid_set);

  std::mt19937 generator;
  std::uniform_int_distribution<uint32_t> distribution(0, num_nodes - 1);

  while (train_set.size() < num_train_set) {
    uint32_t nodeid = distribution(generator);
    if (indptr[nodeid + 1] - indptr[nodeid] > 0 && !bitmap[nodeid]) {
      train_set.push_back(nodeid);
      bitmap[nodeid] = true;
    }
  }

  while (test_set.size() < num_test_set) {
    uint32_t nodeid = distribution(generator);
    if (indptr[nodeid + 1] - indptr[nodeid] > 0 && !bitmap[nodeid]) {
      test_set.push_back(nodeid);
      bitmap[nodeid] = true;
    }
  }

  while (valid_set.size() < num_valid_set) {
    uint32_t nodeid = distribution(generator);
    if (indptr[nodeid + 1] - indptr[nodeid] > 0 && !bitmap[nodeid]) {
      valid_set.push_back(nodeid);
      bitmap[nodeid] = true;
    }
  }

  std::string train_set_path = output_dir + "train_set.bin";
  std::string valid_set_path = output_dir + "valid_set.bin";
  std::string test_set_path = output_dir + "test_set.bin";

  std::ofstream ofs0(train_set_path, std::ofstream::out |
                                         std::ofstream::binary |
                                         std::ofstream::trunc);
  std::ofstream ofs1(valid_set_path, std::ofstream::out |
                                         std::ofstream::binary |
                                         std::ofstream::trunc);
  std::ofstream ofs2(test_set_path, std::ofstream::out | std::ofstream::binary |
                                        std::ofstream::trunc);
  ofs0.write((const char *)train_set.data(),
             train_set.size() * sizeof(uint32_t));
  ofs1.write((const char *)valid_set.data(),
             valid_set.size() * sizeof(uint32_t));
  ofs2.write((const char *)test_set.data(), test_set.size() * sizeof(uint32_t));

  ofs0.close();
  ofs1.close();
  ofs2.close();
}

int main(int argc, char *argv[]) {
  utility::Options::InitOptions("Graph property");
  OPTIONS_PARSE(argc, argv);

  utility::GraphLoader graph_loader(utility::Options::root);
  auto graph = graph_loader.GetGraphDataset(utility::Options::graph);

  generateNodeSet(graph);
}
