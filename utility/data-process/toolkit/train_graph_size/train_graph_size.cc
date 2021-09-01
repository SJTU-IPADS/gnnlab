/**
 * calculate the graph size of using only one batch
 */

#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <cassert>

#include "common/graph_loader.h"
#include "common/options.h"

void TrainSize(utility::GraphPtr dataset) {
  uint32_t *indptr = dataset->indptr;
  uint32_t *indices = dataset->indices;
  uint32_t *train_set = dataset->train_set;
  uint32_t num_train_set = dataset->num_train_set;
  uint32_t num_nodes = dataset->num_nodes;
  uint32_t num_edges = dataset->num_edges;

  // std::vector<std::uint8_t> before(num_nodes, 0);
  volatile uint8_t*  before = new volatile uint8_t [num_nodes]();
  volatile uint8_t*  after = new volatile uint8_t [num_nodes]();

  std::printf("On hop 0: count %12d/%d (%2d%%) nodes\n", num_train_set, num_nodes, num_train_set * 100 / num_nodes);
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
  for (size_t hop_idx = 0; ; hop_idx++) {
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
    std::printf("On hop %lu: count %12lu/%d (%2d%%) nodes\n", hop_idx+1, count, num_nodes, (count * 100) / num_nodes);
    if (last_count == count) break;
    else {
      last_count = count;
    }
  }
}

int main(int argc, char *argv[]) {
  utility::Options::InitOptions("Graph property");
  OPTIONS_PARSE(argc, argv);

  utility::GraphLoader graph_loader(utility::Options::root);
  auto graph = graph_loader.GetGraphDataset(utility::Options::graph);

  TrainSize(graph);

  return 0;
}