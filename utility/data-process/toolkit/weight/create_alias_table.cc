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

std::string output0_filepath = "prob_table.bin";
std::string output1_filepath = "alias_table.bin";

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
  output1_filepath = prefix + output1_filepath;
}

void CreateAliasTable(const uint32_t *indptr, const uint32_t *indices,
                      size_t num_nodes, size_t num_edges,
                      std::vector<float> &prob_table,
                      std::vector<uint32_t> &alias_table) {
#pragma omp parallel for
  for (uint32_t nodeid = 0; nodeid < num_nodes; nodeid++) {
    const uint32_t off = indptr[nodeid];
    const uint32_t len = indptr[nodeid + 1] - off;

    // 1. generate random weight
    std::vector<float> weights(len, 0);
    float weight_sum = 0.0f;

    for (uint32_t i = 0; i < len; i++) {
      weights[i] = static_cast<float>(RandomInt(1, 10));
      weight_sum += weights[i];
    }

    for (uint32_t i = 0; i < len; i++) {
      weights[i] /= weight_sum;
      weights[i] *= len;
    }

    // 2. create alias table
    std::queue<uint32_t> smalls;
    std::queue<uint32_t> larges;

    for (uint32_t i = 0; i < len; i++) {
      if (weights[i] < 1.0) {
        smalls.push(i);
      } else {
        larges.push(i);
      }
    }

    while (!smalls.empty() && !larges.empty()) {
      uint32_t small_idx = smalls.front();
      uint32_t large_idx = larges.front();

      smalls.pop();
      larges.pop();

      prob_table[off + small_idx] = weights[small_idx];
      alias_table[off + small_idx] = indices[off + large_idx];

      weights[large_idx] -= (1 - weights[small_idx]);

      if (weights[large_idx] < 1.0) {
        smalls.push(large_idx);
      } else {
        larges.push(large_idx);
      }
    }

    while (!larges.empty()) {
      uint32_t large_idx = larges.front();
      larges.pop();

      prob_table[off + large_idx] = 1;
    }

    while (!smalls.empty()) {
      uint32_t small_idx = smalls.front();
      smalls.pop();

      prob_table[off + small_idx] = 1;
    }
  }

  std::ofstream ofs0(output0_filepath, std::ofstream::out |
                                           std::ofstream::binary |
                                           std::ofstream::trunc);
  std::ofstream ofs1(output1_filepath, std::ofstream::out |
                                           std::ofstream::binary |
                                           std::ofstream::trunc);

  ofs0.write((const char *)prob_table.data(),
             prob_table.size() * sizeof(float));
  ofs1.write((const char *)alias_table.data(),
             alias_table.size() * sizeof(uint32_t));
  ofs0.close();
  ofs1.close();
}

}  // namespace

int main(int argc, char *argv[]) {
  utility::Options options("Degree generator");
  OPTIONS_PARSE(options, argc, argv);

  utility::GraphLoader graph_loader(options.root);
  auto graph = graph_loader.GetGraphDataset(options.graph);

  uint32_t *indptr = graph->indptr;
  uint32_t *indices = graph->indices;
  size_t num_nodes = graph->num_nodes;
  size_t num_edges = graph->num_edges;

  std::vector<uint32_t> alias_table(graph->num_edges);
  std::vector<float> prob_table(graph->num_edges);

  AddPrefixToFilepath(graph->folder);
  CreateAliasTable(indptr, indices, num_nodes, num_edges, prob_table,
                   alias_table);

  return 0;
}