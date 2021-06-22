#include <omp.h>

#include <iostream>
#include <unordered_map>
#include <unordered_set>

#include "common/graph_loader.h"
#include "common/options.h"

namespace {

size_t num_threads = 24;

}

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
    uint32_t end = indptr[node_src + 1] - start;

    for (uint32_t j = start; j < end; j++) {
      uint32_t node_dst = indices[j];

      adj[node_src].insert(node_dst);
    }
  }

#pragma omp parallel for
  for (uint32_t node_src = 0; node_src < num_nodes; node_src++) {
    uint32_t src_start = indptr[node_src];
    uint32_t src_end = indptr[node_src + 1];
    for (uint32_t j = src_start; j < src_end; j++) {
      uint32_t node_dst = indices[j];
      if (node_dst != node_src && adj[node_dst].count(node_src) > 0) {
        if (node_src != 0) {
          std::cout << "The graph is partial undirected " << std::endl;
        } else {
          std::cout << "The graph is undirected " << std::endl;
        }
        return;
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

#pragma omp parallel for
  for (uint32_t node_src = 0; node_src < num_nodes; node_src++) {
    uint32_t start = indptr[node_src];
    uint32_t end = indptr[node_src + 1] - start;

    for (uint32_t j = start; j < end; j++) {
      uint32_t node_dst = indices[j];

      if (node_dst == node_src) {
        if (node_src != 0) {
          std::cout << "The graph has partial self-loop " << std::endl;
        } else {
          std::cout << "The graph has self-loop " << std::endl;
        }
        return;
      }
    }
  }

  std::cout << "The graph doesn't has self-loop " << std::endl;
}

void HasZeroDegreeNodes(utility::GraphPtr dataset) {
  uint32_t *indptr = indptr;
  uint32_t *indices = indices;
  uint32_t num_nodes = num_nodes;
  uint32_t num_edges = num_edges;
#pragma omp parallel for
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

  omp_set_num_threads(num_threads);

  IsDirected(graph);
  HasSelfLoop(graph);
  HasZeroDegreeNodes(graph);

  return 0;
}