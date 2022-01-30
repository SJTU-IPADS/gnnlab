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
#ifdef __linux__
#include <parallel/algorithm>
#else
#include <algorithm>
#endif
#include <cassert>

#include "common/graph_loader.h"
#include "common/options.h"
#include "common/utils.h"

#include <sys/mman.h>
#include <fcntl.h>
#include <fstream>
#include <cassert>
#include <random>

#include <unistd.h>

namespace {
std::string raw_root = "/graph-learning/data-raw/";
std::string & graph = utility::Options::graph;
size_t num_nodes     = 0;
size_t num_edges     = 0;
size_t feat_dim      = 0;
size_t num_train_set = 0;
size_t num_test_set  = 0;
size_t num_valid_set = 0;
size_t num_class     = 0;

using utility::Check;
using utility::FileExist;
using utility::GraphLoader;

struct coo_wrapper {
  uint32_t * ptr = nullptr;
  inline uint32_t src_of(size_t idx) {
    return ptr[idx * 2];
  }
  inline uint32_t dst_of(size_t idx) {
    return ptr[idx * 2 + 1];
  }
};
coo_wrapper global_coo;
}

void read_meta(std::string graph) {
  std::string graph_root = utility::Options::root + graph + "/";
  Check(FileExist(graph_root + GraphLoader::kMetaFile),
        graph_root + GraphLoader::kMetaFile + " not found");

  std::unordered_map<std::string, size_t> meta;
  std::ifstream meta_file(graph_root + GraphLoader::kMetaFile);
  std::string line;
  while (std::getline(meta_file, line)) {
    std::istringstream iss(line);
    std::vector<std::string> kv{std::istream_iterator<std::string>{iss},
                                std::istream_iterator<std::string>{}};

    if (kv.size() < 2) {
      break;
    }

    meta[kv[0]] = std::stoull(kv[1]);
  }

  Check(meta.count(GraphLoader::kMetaNumNode)     > 0, GraphLoader::kMetaNumNode     + " not exit");
  Check(meta.count(GraphLoader::kMetaNumEdge)     > 0, GraphLoader::kMetaNumEdge     + " not exit");
  Check(meta.count(GraphLoader::kMetaFeatDim)     > 0, GraphLoader::kMetaFeatDim     + " not exit");
  Check(meta.count(GraphLoader::kMetaNumClass)    > 0, GraphLoader::kMetaNumClass    + " not exit");
  Check(meta.count(GraphLoader::kMetaNumTrainSet) > 0, GraphLoader::kMetaNumTrainSet + " not exit");
  Check(meta.count(GraphLoader::kMetaNumTestSet)  > 0, GraphLoader::kMetaNumTestSet  + " not exit");
  Check(meta.count(GraphLoader::kMetaNumValidSet) > 0, GraphLoader::kMetaNumValidSet + " not exit");

  num_nodes     = meta[GraphLoader::kMetaNumNode];
  num_edges     = meta[GraphLoader::kMetaNumEdge];
  feat_dim      = meta[GraphLoader::kMetaFeatDim];
  num_class     = meta[GraphLoader::kMetaNumClass];
  num_train_set = meta[GraphLoader::kMetaNumTrainSet];
  num_test_set  = meta[GraphLoader::kMetaNumTestSet];
  num_valid_set = meta[GraphLoader::kMetaNumValidSet];
}

void load_coo(std::string file) {
  if (!utility::FileExist(file)) {
    return;
  }

  int fd;
  struct stat st;
  size_t nbytes;
  void *ret;

  fd = open(file.c_str(), O_RDONLY, 0);
  stat(file.c_str(), &st);
  nbytes = st.st_size;

  ret = mmap(NULL, nbytes, PROT_READ, MAP_SHARED, fd, 0);
  mlock(ret, nbytes);
  close(fd);
  assert(nbytes / sizeof(uint32_t) / 2 == num_edges);
  std::cout << "detected " << num_edges << " edges\n";
  global_coo.ptr = static_cast<uint32_t*>(ret);
}

void check_max_node_id() {
  uint32_t node_id = 0;
  for (size_t edge_id = 0; edge_id < num_edges; edge_id++) {
    node_id = std::max(global_coo.dst_of(edge_id), node_id);
    node_id = std::max(global_coo.src_of(edge_id), node_id);
  }
  std::cout << "max node id is " << node_id << "\n";
  if (node_id >= num_nodes) {
    Check(false, "max node id >= num nodes");
  }
}

void coo_to_csc(void *coo_ptr, std::vector<uint32_t> &indptr, std::vector<uint32_t> &indices) {
  std::vector<std::pair<uint32_t, uint32_t>> dst_to_src_edge_list(num_edges);
  for (size_t edge_id = 0; edge_id < num_edges; edge_id++) {
    dst_to_src_edge_list[edge_id].first = global_coo.dst_of(edge_id);
    dst_to_src_edge_list[edge_id].second = global_coo.src_of(edge_id);
  }
#ifdef __linux__
  __gnu_parallel::sort(dst_to_src_edge_list.begin(), dst_to_src_edge_list.end(),
                       std::less<std::pair<uint32_t, uint32_t>>());
#else
  std::sort(dst_to_src_edge_list.begin(), dst_to_src_edge_list.end(),
            std::less<std::pair<uint32_t, uint32_t>>());
#endif
  indptr.resize(num_nodes + 1);
  indices.resize(num_edges);
  size_t cur_edge_id = 0;
  for (uint32_t dst_id = 0; dst_id < num_nodes; dst_id++) {
    indptr[dst_id] = cur_edge_id;
    while (dst_to_src_edge_list[cur_edge_id].first == dst_id) {
      indices[cur_edge_id] = dst_to_src_edge_list[cur_edge_id].second;
      cur_edge_id++;
    }
  }
  indptr[num_nodes] = cur_edge_id;
  assert(cur_edge_id == num_edges);
}

void store_csc(std::vector<uint32_t> &indptr, std::vector<uint32_t> &indices) {
  std::string output_dir = utility::Options::root + graph + "/";
  std::string indptr_path = output_dir + "indptr.bin";
  std::string indices_path = output_dir + "indices.bin";
  std::ofstream ofs0(indptr_path, std::ofstream::out | std::ofstream::binary |
                                      std::ofstream::trunc);
  std::ofstream ofs1(indices_path, std::ofstream::out | std::ofstream::binary |
                                       std::ofstream::trunc);

  ofs0.write((const char *)indptr.data(), (num_nodes + 1) * sizeof(uint32_t));
  ofs1.write((const char *)indices.data(), num_edges * sizeof(uint32_t));

  ofs0.close();
  ofs1.close();
}

void generateNodeSet(std::vector<uint32_t> &indptr) {
  std::string output_dir = utility::Options::root + graph + "/";
  std::vector<bool> bitmap(num_nodes, false);
  std::vector<uint32_t> train_set;
  std::vector<uint32_t> test_set;
  std::vector<uint32_t> valid_set;

  train_set.reserve(num_train_set);
  test_set.reserve(num_test_set);
  valid_set.reserve(num_valid_set);

  std::mt19937 generator;
  std::uniform_int_distribution<uint32_t> distribution(0, num_nodes - 1);

  while (train_set.size() < num_train_set) {
    uint32_t nodeid = distribution(generator);
    if (indptr[nodeid + 1] - indptr[nodeid] > 0 && !bitmap[nodeid]) {
      train_set.push_back(nodeid);
      bitmap[nodeid] = true;
    }
  }

  while (test_set.size() < num_test_set) {
    uint32_t nodeid = distribution(generator);
    if (indptr[nodeid + 1] - indptr[nodeid] > 0 && !bitmap[nodeid]) {
      test_set.push_back(nodeid);
      bitmap[nodeid] = true;
    }
  }

  while (valid_set.size() < num_valid_set) {
    uint32_t nodeid = distribution(generator);
    if (indptr[nodeid + 1] - indptr[nodeid] > 0 && !bitmap[nodeid]) {
      valid_set.push_back(nodeid);
      bitmap[nodeid] = true;
    }
  }

  std::string train_set_path = output_dir + "train_set.bin";
  std::string valid_set_path = output_dir + "valid_set.bin";
  std::string test_set_path = output_dir + "test_set.bin";

  std::ofstream ofs0(train_set_path, std::ofstream::out |
                                         std::ofstream::binary |
                                         std::ofstream::trunc);
  std::ofstream ofs1(valid_set_path, std::ofstream::out |
                                         std::ofstream::binary |
                                         std::ofstream::trunc);
  std::ofstream ofs2(test_set_path, std::ofstream::out | std::ofstream::binary |
                                        std::ofstream::trunc);
  ofs0.write((const char *)train_set.data(),
             train_set.size() * sizeof(uint32_t));
  ofs1.write((const char *)valid_set.data(),
             valid_set.size() * sizeof(uint32_t));
  ofs2.write((const char *)test_set.data(), test_set.size() * sizeof(uint32_t));

  ofs0.close();
  ofs1.close();
  ofs2.close();
}

int main(int argc, char* argv[]) {
  utility::Options::InitOptions("COO to Dataset");

  OPTIONS_PARSE(argc, argv);
  read_meta(graph);
  load_coo(raw_root + graph + "/" + "coo.bin");
  check_max_node_id();
  std::vector<uint32_t> indptr, indices;
  coo_to_csc(global_coo.ptr, indptr, indices);
  store_csc(indptr, indices);
  generateNodeSet(indptr);
}