#include <omp.h>

#include <iostream>

#include "common/graph_loader.h"
#include "common/options.h"

void IsDirected(utility::GraphPtr dataset) {
  uint32_t *indptr = dataset->indptr;
  uint32_t *indices = dataset->indices;
  uint32_t num_nodes = dataset->num_nodes;
  uint32_t num_edges = dataset->num_edges;

  for (uint32_t node_src = 0; node_src < 10 && node_src < num_nodes;
       node_src++) {
    uint32_t src_start = indptr[node_src];
    uint32_t src_end = indptr[node_src + 1];
    for (uint32_t j = src_start; j < src_end; j++) {
      uint32_t node_dst = indices[j];

      uint32_t dst_start = indptr[node_dst];
      uint32_t dst_end = indptr[node_dst + 1];

      for (uint32_t k = dst_start; k < dst_end; k++) {
        if (indices[k] == node_src) {
          std::cout << "The graph is undirected " << std::endl;
          return;
        }
      }
    }
  }
  std::cout << "The graph is directed " << std::endl;
}

void HasSelfLoop(utility::GraphPtr dataset) {
  uint32_t *indptr = dataset->indptr;
  uint32_t *indices = dataset->indices;
  uint32_t num_nodes = dataset->num_nodes;
  uint32_t num_edges = dataset->num_edges;

  uint32_t node_src = 0;
  uint32_t start = indptr[node_src];
  uint32_t end = indptr[node_src + 1] - start;

  for (uint32_t j = start; j < end; j++) {
    uint32_t node_dst = indices[j];

    if (node_dst == node_src) {
      std::cout << "The graph has self-loop " << std::endl;
      return;
    }
  }

  std::cout << "The graph doesn't has self-loop " << std::endl;
}

void HasZeroDegreeNodes(utility::GraphPtr dataset) {
  uint32_t *indptr = indptr;
  uint32_t *indices = indices;
  uint32_t num_nodes = num_nodes;
  uint32_t num_edges = num_edges;

  for (uint32_t i = 0; i < num_nodes; i++) {
    uint32_t len = indptr[i + 1] - indptr[i];
    if (len == 0) {
      std::cout << "The graph has zero-degree nodes" << std::endl;
      return;
    }
  }
  std::cout << "The graph doesn't has zero-degree nodes" << std::endl;
}

int main(int argc, char *argv[]) {
  utility::Options options("Check whether the graph is undirected");
  OPTIONS_PARSE(options, argc, argv);

  utility::GraphLoader graph_loader(options.root);
  auto graph = graph_loader.GetGraphDataset(options.graph);

  IsDirected(graph);
  HasSelfLoop(graph);
  HasZeroDegreeNodes(graph);

  return 0;
}