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

std::string output0_filepath = "prob_prefix_table.bin";
std::shared_ptr<utility::DegreeInfo> degree_info;

#define WEIGHT_POLICY_TYPES( F ) \
  F(kDefault) \
  F(kInverseBothDegreeRand) \
  F(kInverseSrcDegreeRand) \
  F(kSrcSuffix)

#define F(name) name,
enum WeightPolicy {WEIGHT_POLICY_TYPES( F ) kNumItems };
#undef F
WeightPolicy weight_policy = kSrcSuffix;

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
}\
float SrcSuffix(uint32_t src, uint32_t dst) {
  if (degree_info->out_degrees[src] < 10) return 100;
  else return 1;
  // in paper
  // if (src % 10 < 3 && degree_info->out_degrees[src] < 15) return 100;
  // else if (src % 10 < 3) return 10;
  // else if (degree_info->out_degrees[src] < 15) return 10;
  // else return 1;
}

void CreateProbPrefixTable(const uint32_t *indptr, const uint32_t *indices,
                      size_t num_nodes, size_t num_edges,
                      std::vector<float> &prob_prefix_table) {
#pragma omp parallel for
  for (uint32_t nodeid = 0; nodeid < num_nodes; nodeid++) {
    const uint32_t off = indptr[nodeid];
    const uint32_t len = indptr[nodeid + 1] - off;

    // 1. generate random weight
    float weight_sum = 0.0f;

    for (uint32_t i = 0; i < len; i++) {
      uint32_t dst = nodeid, src = indices[off+i];
      float weight;
      switch(weight_policy) {
        case kDefault:
          weight = static_cast<float>(RandomInt(1, 10)); break;
        case kInverseBothDegreeRand:
          weight = InverseBothDegreeRand(src, dst); break;
        case kInverseSrcDegreeRand:
          weight = InverseSrcDegreeRand(src, dst); break;
        case kSrcSuffix:
          weight = SrcSuffix(src, dst); break;
        default:
          utility::Check(false);
      }
      weight_sum += weight;
      prob_prefix_table[off + i] = weight_sum;
    }
  }

  std::ofstream ofs0(output0_filepath, std::ofstream::out |
                                           std::ofstream::binary |
                                           std::ofstream::trunc);

  ofs0.write((const char *)prob_prefix_table.data(),
             prob_prefix_table.size() * sizeof(float));
  ofs0.close();
}

}  // namespace

int main(int argc, char *argv[]) {
  std::string policy_str = "kSrcSuffix";
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

  std::vector<float> prob_table(graph->num_edges);

  AddPrefixToFilepath(graph->folder);
  CreateProbPrefixTable(indptr, indices, num_nodes, num_edges, prob_table);

  return 0;
}