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

#include <omp.h>

#include <random>
#include <vector>

#include "common/graph_loader.h"
#include "common/options.h"
#include "common/utils.h"

int main(int argc, char *argv[]) {
  utility::Options::InitOptions("Degree generator");
  OPTIONS_PARSE(argc, argv);

  utility::GraphLoader graph_loader(utility::Options::root);
  auto graph = graph_loader.GetGraphDataset(utility::Options::graph);

  uint32_t *indptr = graph->indptr;
  uint32_t *indices = graph->indices;
  size_t num_nodes = graph->num_nodes;
  size_t num_edges = graph->num_edges;
  float *feat = graph->feature;
  size_t dim = graph->feat_dim;

  size_t num_runs = 10;
  size_t num_inputs = 8000;

  std::vector<std::vector<int>> inputs(num_runs, std::vector<int>(num_inputs));
  std::vector<std::vector<float>> outputs(num_runs,
                                          std::vector<float>(num_inputs * dim));

  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_int_distribution<std::mt19937::result_type> dist(
      0, num_nodes - 1);  // distribution in range [1, 6]

  for (size_t i = 0; i < num_runs; i++) {
    for (size_t j = 0; j < num_inputs; j++) {
      inputs[i][j] = dist(rng);
    }
  }

  for (size_t i = 0; i < num_runs; i++) {
    utility::Timer t;
    // #pragma omp parallel for num_threads(1)
    for (size_t j = 0; j < num_inputs; j++) {
      // #pragma omp simd
      for (size_t k = 0; k < dim; k++) {
        outputs[i][j * dim + k] = feat[inputs[i][j] * dim + k];
      }

      // std::cout << "OMP thread id %d" << omp_get_thread_num() << std::endl;
    }

    double d = t.Passed();
    std::cout << "[Run " << i << "] " << d << " secs" << std::endl;
  }
}
