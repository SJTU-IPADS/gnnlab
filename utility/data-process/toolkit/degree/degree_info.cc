#include <fcntl.h>
#include <omp.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef __linux__
#include <parallel/algorithm>
#else
#include <algorithm>
#endif

#include "common/graph_loader.h"
#include "common/options.h"

namespace {

std::string out0_filepath = "degrees.txt";
std::string out1_filepath = "out_degrees.bin";
std::string out2_filepath = "in_degrees.bin";
std::string out3_filepath = "in_degree_frequency.txt";
std::string out4_filepath = "out_degree_frequency.txt";
std::string out5_filepath = "sorted_nodes_by_in_degree.bin";

void AddPrefixToFilepath(std::string prefix) {
  if (prefix.back() != '/') {
    prefix += '/';
  }

  out0_filepath = prefix + out0_filepath;
  out1_filepath = prefix + out1_filepath;
  out2_filepath = prefix + out2_filepath;
  out3_filepath = prefix + out3_filepath;
  out4_filepath = prefix + out4_filepath;
  out5_filepath = prefix + out5_filepath;
};

void getNodeDegrees(const uint32_t *indptr, const uint32_t *indices,
                    size_t num_nodes, std::vector<uint32_t> &in_degrees,
                    std::vector<uint32_t> &out_degrees, size_t num_threads) {
  std::vector<std::vector<uint32_t>> in_degrees_per_thread(
      num_threads, std::vector<uint32_t>(num_nodes, 0));

#pragma omp parallel for
  for (uint32_t i = 0; i < num_nodes; i++) {
    uint32_t len = indptr[i + 1] - indptr[i];
    uint32_t off = indptr[i];
    out_degrees[i] = len;

    uint32_t thread_idx = omp_get_thread_num();
    for (uint32_t k = 0; k < len; k++) {
      in_degrees_per_thread[thread_idx][indices[off + k]]++;
    }
  }
#pragma omp parallel for
  for (uint32_t i = 0; i < num_nodes; i++) {
    for (uint32_t k = 0; k < num_threads; k++) {
      in_degrees[i] += in_degrees_per_thread[k][i];
    }
  }
}

void degreesToFile(const std::vector<uint32_t> &in_degrees,
                   const std::vector<uint32_t> &out_degrees, size_t num_nodes) {
  std::ofstream ofs0(out0_filepath, std::ofstream::out | std::ofstream::trunc);
  std::ofstream ofs1(out1_filepath, std::ofstream::out | std::ofstream::binary |
                                        std::ofstream::trunc);
  std::ofstream ofs2(out2_filepath, std::ofstream::out | std::ofstream::binary |
                                        std::ofstream::trunc);

  for (uint32_t i = 0; i < num_nodes; i++) {
    ofs0 << i << " " << out_degrees[i] << " " << in_degrees[i] << "\n";
  }

  ofs1.write((const char *)out_degrees.data(),
             out_degrees.size() * sizeof(uint32_t));
  ofs2.write((const char *)in_degrees.data(),
             in_degrees.size() * sizeof(uint32_t));
  ofs0.close();
  ofs1.close();
  ofs2.close();
}

void degreeFrequencyToFile(const std::vector<uint32_t> &in_degrees,
                           const std::vector<uint32_t> &out_degrees,
                           size_t num_nodes, size_t num_edges) {
  std::unordered_map<uint32_t, size_t> indegree_frequency_map;
  std::unordered_map<uint32_t, size_t> outdegree_frequency_map;

  std::vector<std::pair<uint32_t, size_t>> indegree_frequency;
  std::vector<std::pair<uint32_t, size_t>> outdegree_frequency;

  double indegree_node_frequency_percentage_prefix_sum = 0;
  double outdegree_node_frequency_percentage_prefix_sum = 0;
  double indegree_edge_frequency_percentage_prefix_sum = 0;
  double outdegree_edge_frequency_percentage_prefix_sum = 0;

  for (size_t i = 0; i < in_degrees.size(); i++) {
    indegree_frequency_map[in_degrees[i]]++;
  }

  for (size_t i = 0; i < out_degrees.size(); i++) {
    outdegree_frequency_map[out_degrees[i]]++;
  }

  for (auto &p : indegree_frequency_map) {
    indegree_frequency.emplace_back(p.first, p.second);
  }

  for (auto &p : outdegree_frequency_map) {
    outdegree_frequency.emplace_back(p.first, p.second);
  }

#ifdef __linux__
  __gnu_parallel::sort(indegree_frequency.begin(), indegree_frequency.end(),
                       std::greater<std::pair<uint32_t, size_t>>());
  __gnu_parallel::sort(outdegree_frequency.begin(), outdegree_frequency.end(),
                       std::greater<std::pair<uint32_t, size_t>>());
#else
  std::sort(indegree_frequency.begin(), indegree_frequency.end(),
            std::greater<std::pair<uint32_t, size_t>>());
  std::sort(outdegree_frequency.begin(), outdegree_frequency.end(),
            std::greater<std::pair<uint32_t, size_t>>());
#endif

  std::ofstream ofs3(out3_filepath, std::ofstream::out | std::ofstream::trunc);
  std::ofstream ofs4(out4_filepath, std::ofstream::out | std::ofstream::trunc);

  for (auto &p : indegree_frequency) {
    uint32_t degree = p.first;
    size_t frequency = p.second;

    double node_percentage =
        static_cast<double>(frequency) / static_cast<double>(num_nodes);
    double edge_percentage = static_cast<double>(degree * frequency) /
                             static_cast<double>(num_edges);
    indegree_node_frequency_percentage_prefix_sum += node_percentage;
    indegree_edge_frequency_percentage_prefix_sum += edge_percentage;

    ofs3 << degree << " " << frequency << " " << node_percentage << " "
         << indegree_node_frequency_percentage_prefix_sum << " "
         << edge_percentage << " "
         << indegree_edge_frequency_percentage_prefix_sum << "\n";
  }

  for (auto &p : outdegree_frequency) {
    uint32_t degree = p.first;
    size_t frequency = p.second;

    double node_percentage =
        static_cast<double>(frequency) / static_cast<double>(num_nodes);
    double edge_percentage = static_cast<double>(degree * frequency) /
                             static_cast<double>(num_edges);
    outdegree_node_frequency_percentage_prefix_sum += node_percentage;
    outdegree_edge_frequency_percentage_prefix_sum += edge_percentage;

    ofs4 << degree << " " << frequency << " " << node_percentage << " "
         << outdegree_node_frequency_percentage_prefix_sum << " "
         << edge_percentage << " "
         << outdegree_edge_frequency_percentage_prefix_sum << "\n";
  }

  ofs3.close();
  ofs4.close();
}

void sortedNodesToFile(const std::vector<uint32_t> &in_degrees) {
  std::vector<std::pair<uint32_t, uint32_t>> in_degrees_ids_list;
  for (uint32_t i = 0; i < in_degrees.size(); i++) {
    in_degrees_ids_list.emplace_back(in_degrees[i], i);
  }

#ifdef __linux__
  __gnu_parallel::sort(in_degrees_ids_list.begin(), in_degrees_ids_list.end(),
                       std::greater<std::pair<uint32_t, uint32_t>>());
#else
  std::sort(in_degrees_ids_list.begin(), in_degrees_ids_list.end(),
            std::greater<std::pair<uint32_t, uint32_t>>());
#endif

  std::vector<uint32_t> nodes_sorted_by_in_degree;

  for (size_t i = 0; i < in_degrees_ids_list.size(); i++) {
    nodes_sorted_by_in_degree.push_back(in_degrees_ids_list[i].second);
  }

  std::ofstream ofs5(out5_filepath, std::ofstream::out | std::ofstream::binary |
                                        std::ofstream::trunc);

  ofs5.write((const char *)nodes_sorted_by_in_degree.data(),
             nodes_sorted_by_in_degree.size() * sizeof(uint32_t));

  ofs5.close();
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
  std::vector<uint32_t> in_degrees(num_nodes, 0);
  std::vector<uint32_t> out_degrees(num_nodes, 0);

  AddPrefixToFilepath(graph->folder);
  getNodeDegrees(indptr, indices, num_nodes, in_degrees, out_degrees,
                 options.num_threads);
  degreesToFile(in_degrees, out_degrees, num_nodes);
  degreeFrequencyToFile(in_degrees, out_degrees, num_nodes, num_edges);
  sortedNodesToFile(in_degrees);
}
