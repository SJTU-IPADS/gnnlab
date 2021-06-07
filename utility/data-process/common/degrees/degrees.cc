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

std::string dataset = "papers100M";
// std::string dataset = "com-friendster";
std::unordered_map<std::string, size_t> dataset2nodes = {
    {"com-friendster", 65608366}, {"papers100M", 111059956}};
std::unordered_map<std::string, size_t> dataset2edges = {
    {"com-friendster", 1871675501}, {"papers100M", 1726745828}};

size_t num_nodes = dataset2nodes[dataset];
size_t num_edges = dataset2edges[dataset];
size_t num_threads = 24;

std::string dataroot = "/graph-learning/samgraph/";
std::string prefix = dataroot + dataset + "/";

std::string indptr_filepath = prefix + "indptr.bin";
std::string indices_filepath = prefix + "indices.bin";
std::string out0_filepath = prefix + "degrees.txt";
std::string out1_filepath = prefix + "out_degrees.bin";
std::string out2_filepath = prefix + "in_degrees.bin";
std::string out3_filepath = prefix + "in_degree_frequency.txt";
std::string out4_filepath = prefix + "out_degree_frequency.txt";
std::string out5_filepath = prefix + "sorted_nodes_by_in_degree.bin";

void loadGraph(uint32_t **indptr, uint32_t **indices) {
  int fd;
  struct stat st;
  size_t nbytes;

  fd = open(indptr_filepath.c_str(), O_RDONLY, 0);
  stat(indptr_filepath.c_str(), &st);
  nbytes = st.st_size;

  if (nbytes == 0) {
    std::cout << "Reading file error: " << indptr_filepath << std::endl;
    exit(1);
  }

  *indptr =
      (uint32_t *)mmap(NULL, nbytes, PROT_READ, MAP_SHARED | MAP_FILE, fd, 0);
  mlock(indptr, nbytes);
  close(fd);

  fd = open(indices_filepath.c_str(), O_RDONLY, 0);
  stat(indices_filepath.c_str(), &st);
  nbytes = st.st_size;
  if (nbytes == 0) {
    std::cout << "Reading file error: " << indices_filepath << std::endl;
    exit(1);
  }

  *indices =
      (uint32_t *)mmap(NULL, nbytes, PROT_READ, MAP_SHARED | MAP_FILE, fd, 0);
  mlock(indices, nbytes);
  close(fd);
}

void getNodeDegrees(const uint32_t *indptr, const uint32_t *indices,
                    std::vector<uint32_t> &in_degrees,
                    std::vector<uint32_t> &out_degrees) {
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
                   const std::vector<uint32_t> &out_degrees) {
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
                           const std::vector<uint32_t> &out_degrees) {
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

int main() {
  uint32_t *indptr;
  uint32_t *indices;
  std::vector<uint32_t> in_degrees(num_nodes, 0);
  std::vector<uint32_t> out_degrees(num_nodes, 0);

  omp_set_num_threads(num_threads);

  loadGraph(&indptr, &indices);
  getNodeDegrees(indptr, indices, in_degrees, out_degrees);
  degreesToFile(in_degrees, out_degrees);
  degreeFrequencyToFile(in_degrees, out_degrees);
  sortedNodesToFile(in_degrees);
}
