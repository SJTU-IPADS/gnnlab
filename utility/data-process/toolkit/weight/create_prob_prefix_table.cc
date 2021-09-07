#include <fstream>
#include <iostream>
#include <queue>
#include <random>
#include <string>
#include <vector>

#include "common/graph_loader.h"
#include "common/options.h"
#include "common/utils.h"

namespace {

std::string output0_filepath = "prob_prefix_table.bin";

uint32_t RandomInt(const uint32_t &min, const uint32_t &max) {
  static thread_local std::mt19937 generator;
  std::uniform_int_distribution<uint32_t> distribution(min, max);
  return distribution(generator);
}

void AddPrefixToFilepath(std::string prefix) {
  if (prefix.back() != '/') {
    prefix += '/';
  }

  output0_filepath = prefix + output0_filepath;
}

void CreateProbPrefixTable(const uint32_t *indptr, const uint32_t *indices,
                      size_t num_nodes, size_t num_edges,
                      std::vector<uint32_t> &prob_table) {
#pragma omp parallel for
  for (uint32_t nodeid = 0; nodeid < num_nodes; nodeid++) {
    const uint32_t off = indptr[nodeid];
    const uint32_t len = indptr[nodeid + 1] - off;

    // 1. generate random weight
    uint32_t weight_sum = 0;

    for (uint32_t i = 0; i < len; i++) {
      auto weight = RandomInt(1, 100);
      weight_sum += weight;
      prob_table[off + i] = weight_sum;
    }

  }

  std::ofstream ofs0(output0_filepath, std::ofstream::out |
                                           std::ofstream::binary |
                                           std::ofstream::trunc);

  ofs0.write((const char *)prob_table.data(),
             prob_table.size() * sizeof(uint32_t));
  ofs0.close();
}

}  // namespace

int main(int argc, char *argv[]) {
  utility::Options::InitOptions("Graph property");
  OPTIONS_PARSE(argc, argv);

  utility::GraphLoader graph_loader(utility::Options::root);
  auto graph = graph_loader.GetGraphDataset(utility::Options::graph);

  uint32_t *indptr = graph->indptr;
  uint32_t *indices = graph->indices;
  size_t num_nodes = graph->num_nodes;
  size_t num_edges = graph->num_edges;

  std::vector<uint32_t> prob_table(graph->num_edges);

  AddPrefixToFilepath(graph->folder);
  CreateProbPrefixTable(indptr, indices, num_nodes, num_edges, prob_table);

  return 0;
}