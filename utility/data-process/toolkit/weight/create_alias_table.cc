#include <fstream>
#include <iostream>
#include <queue>
#include <random>
#include <string>
#include <vector>
#include <unordered_map>

#include "common/graph_loader.h"
#include "common/options.h"
#include "common/utils.h"

namespace {

std::string output0_filepath = "prob_table.bin";
std::string output1_filepath = "alias_table.bin";
std::shared_ptr<utility::DegreeInfo> degree_info;

#define WEIGHT_POLICY_TYPES( F ) \
  F(kDefault) \
  F(kInverseBothDegreeRand) \
  F(kInverseSrcDegreeRand)

#define F(name) name,
enum WeightPolicy {WEIGHT_POLICY_TYPES( F ) kNumItems };
#undef F
WeightPolicy weight_policy = kDefault;

#define F(name) {#name,name},
std::unordered_map<std::string, WeightPolicy> policy_str_to_int = {
WEIGHT_POLICY_TYPES( F )
};
#undef F

void InitPolicy(std::string policy_str) {
  if (policy_str_to_int.find(policy_str) == policy_str_to_int.end()) {
    utility::Check(false, "wrong policy " + policy_str);
  }
  weight_policy = policy_str_to_int[policy_str];
}

uint32_t RandomInt(const uint32_t &min, const uint32_t &max) {
  static thread_local std::random_device dev;
  static thread_local std::mt19937 generator(dev());
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

float InverseSrcDegreeRand(uint32_t src, uint32_t dst) {
  uint32_t src_out_deg = degree_info->out_degrees[src];
  // return 1.0 / RandomInt(1, src_out_deg);
  return 1.0 / src_out_deg;
}
float InverseBothDegreeRand(uint32_t src, uint32_t dst) {
  uint32_t src_out_deg = degree_info->out_degrees[src];
  uint32_t dst_in_deg = degree_info->in_degrees[dst];
  return 1.0 / RandomInt(1, std::max(src_out_deg, dst_in_deg));
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
      uint32_t dst = nodeid, src = indices[off+i];
      switch(weight_policy) {
        case kDefault:
          weights[i] = static_cast<float>(RandomInt(1, 10)); break;
        case kInverseBothDegreeRand:
          weights[i] = InverseBothDegreeRand(src, dst); break;
        case kInverseSrcDegreeRand:
          weights[i] = InverseSrcDegreeRand(src, dst); break;
        default:
          utility::Check(false);
      }
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
  std::string policy_str;
  utility::Options::CustomOption("-P,--policy", policy_str);
  utility::Options::InitOptions("Graph property");
  OPTIONS_PARSE(argc, argv);
  InitPolicy(policy_str);

  utility::GraphLoader graph_loader(utility::Options::root);
  auto graph = graph_loader.GetGraphDataset(utility::Options::graph);
  degree_info = utility::DegreeInfo::GetDegrees(graph);

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