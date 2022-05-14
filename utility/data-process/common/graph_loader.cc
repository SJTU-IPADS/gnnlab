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

#include "graph_loader.h"

#include <fcntl.h>
#include <omp.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "options.h"
#include "utils.h"

namespace utility {

const std::string GraphLoader::kMetaFile = "meta.txt";
const std::string GraphLoader::kFeatFile = "feat.bin";
const std::string GraphLoader::kLabelFile = "label.bin";
const std::string GraphLoader::kIndptrFile = "indptr.bin";
const std::string GraphLoader::kIndicesFile = "indices.bin";
const std::string GraphLoader::kTrainSetFile = "train_set.bin";
const std::string GraphLoader::kTestSetFile = "test_set.bin";
const std::string GraphLoader::kValidSetFile = "valid_set.bin";
const std::string GraphLoader::kIndptr64File = "indptr64.bin";
const std::string GraphLoader::kIndices64File = "indices64.bin";
const std::string GraphLoader::kTrainSet64File = "train_set64.bin";
const std::string GraphLoader::kTestSet64File = "test_set64.bin";
const std::string GraphLoader::kValidSet64File = "valid_set64.bin";

const std::string GraphLoader::kMetaFeatDataType = "FEAT_DATA_TYPE";
const std::string GraphLoader::kMetaNumNode = "NUM_NODE";
const std::string GraphLoader::kMetaNumEdge = "NUM_EDGE";
const std::string GraphLoader::kMetaFeatDim = "FEAT_DIM";
const std::string GraphLoader::kMetaNumClass = "NUM_CLASS";
const std::string GraphLoader::kMetaNumTrainSet = "NUM_TRAIN_SET";
const std::string GraphLoader::kMetaNumTestSet = "NUM_TEST_SET";
const std::string GraphLoader::kMetaNumValidSet = "NUM_VALID_SET";

Graph::Graph()
    : indptr(nullptr),
      indices(nullptr),
      train_set(nullptr),
      test_set(nullptr),
      valid_set(nullptr),
      indptr64(nullptr),
      indices64(nullptr),
      train_set64(nullptr),
      test_set64(nullptr),
      valid_set64(nullptr) {}

Graph::~Graph() {
  // if (indptr) {
  //   int ret = munmap(indptr, sizeof(uint32_t) * (num_nodes + 1));
  //   Check(ret == 0, "munmap indptr error");
  // }

  // if (indices) {
  //   int ret = munmap(indices, sizeof(uint32_t) * num_edges);
  //   Check(ret == 0, "munmap indices error");
  // }
}

void *Graph::LoadDataFromFile(std::string file, const size_t expected_nbytes) {
  if (!FileExist(file)) {
    return nullptr;
  }

  int fd;
  struct stat st;
  size_t nbytes;
  void *ret;

  fd = open(file.c_str(), O_RDONLY, 0);
  stat(file.c_str(), &st);
  nbytes = st.st_size;

  Check(nbytes == expected_nbytes, "Reading file error: " + file);

  ret = mmap(NULL, nbytes, PROT_READ, MAP_SHARED | MAP_FILE, fd, 0);
  mlock(ret, nbytes);
  close(fd);

  return ret;
}

std::shared_ptr<DegreeInfo> DegreeInfo::GetDegrees(GraphPtr &graph) {
  size_t num_nodes = graph->num_nodes;
  size_t num_threads = Options::num_threads;
  uint32_t *indptr = graph->indptr;
  uint32_t *indices = graph->indices;

  auto info = std::make_shared<DegreeInfo>();
  std::vector<uint32_t> &in_degrees = info->in_degrees;
  std::vector<uint32_t> &out_degrees = info->out_degrees;

  in_degrees.resize(num_nodes);
  out_degrees.resize(num_nodes);

  // The graph is CSC-format
  std::vector<std::vector<uint32_t>> out_degrees_per_thread(
      num_threads, std::vector<uint32_t>(num_nodes, 0));

#pragma omp parallel for
  for (uint32_t i = 0; i < num_nodes; i++) {
    uint32_t len = indptr[i + 1] - indptr[i];
    uint32_t off = indptr[i];

    in_degrees[i] = len;

    uint32_t thread_idx = omp_get_thread_num();
    for (uint32_t k = 0; k < len; k++) {
      out_degrees_per_thread[thread_idx][indices[off + k]]++;
    }
  }

#pragma omp parallel for
  for (uint32_t i = 0; i < num_nodes; i++) {
    for (uint32_t k = 0; k < num_threads; k++) {
      out_degrees[i] += out_degrees_per_thread[k][i];
    }
  }

  return info;
}

GraphLoader::GraphLoader(std::string root) {
  if (root.back() != '/') {
    _root = root + '/';
  } else {
    _root = root;
  }
}

GraphPtr GraphLoader::GetGraphDataset(std::string graph, bool is64type) {
  auto dataset = std::make_shared<Graph>();

  dataset->folder = _root + graph + '/';

  std::cout << "Loading graph data from " << dataset->folder << std::endl;

  Check(FileExist(dataset->folder + kMetaFile),
        dataset->folder + kMetaFile + " not found");

  std::unordered_map<std::string, size_t> meta;
  std::ifstream meta_file(dataset->folder + kMetaFile);
  std::string line;
  while (std::getline(meta_file, line)) {
    std::istringstream iss(line);
    std::vector<std::string> kv{std::istream_iterator<std::string>{iss},
                                std::istream_iterator<std::string>{}};

    if (kv.size() < 2) {
      break;
    }

    if (kv[0] == kMetaFeatDataType) {
      // toolkit does not use feature type for now?
      continue;
    } else {
      meta[kv[0]] = std::stoull(kv[1]);
    }
  }

  Check(meta.count(kMetaNumNode) > 0, kMetaNumNode + " not exist");
  Check(meta.count(kMetaNumEdge) > 0, kMetaNumEdge + " not exist");
  Check(meta.count(kMetaFeatDim) > 0, kMetaFeatDim + " not exist");
  Check(meta.count(kMetaNumClass) > 0, kMetaNumClass + " not exist");
  Check(meta.count(kMetaNumTrainSet) > 0, kMetaNumTrainSet + " not exist");
  Check(meta.count(kMetaNumTestSet) > 0, kMetaNumTestSet + " not exist");
  Check(meta.count(kMetaNumValidSet) > 0, kMetaNumValidSet + " not exist");
  Check(meta.count(kMetaFeatDim) > 0, kMetaFeatDim + " not exist");

  dataset->num_nodes = meta[kMetaNumNode];
  dataset->num_edges = meta[kMetaNumEdge];
  dataset->num_train_set = meta[kMetaNumTrainSet];
  dataset->num_valid_set = meta[kMetaNumValidSet];
  dataset->num_test_set = meta[kMetaNumTestSet];
  dataset->feat_dim = meta[kMetaFeatDim];

  if (!is64type) {
    dataset->indptr = static_cast<uint32_t *>(
        Graph::LoadDataFromFile(dataset->folder + kIndptrFile,
                                (meta[kMetaNumNode] + 1) * sizeof(uint32_t)));
    dataset->indices = static_cast<uint32_t *>(Graph::LoadDataFromFile(
        dataset->folder + kIndicesFile, meta[kMetaNumEdge] * sizeof(uint32_t)));
    dataset->train_set = static_cast<uint32_t *>(
        Graph::LoadDataFromFile(dataset->folder + kTrainSetFile,
                                (meta[kMetaNumTrainSet]) * sizeof(uint32_t)));
    dataset->test_set = static_cast<uint32_t *>(
        Graph::LoadDataFromFile(dataset->folder + kTestSetFile,
                                meta[kMetaNumTestSet] * sizeof(uint32_t)));
    dataset->valid_set = static_cast<uint32_t *>(
        Graph::LoadDataFromFile(dataset->folder + kValidSetFile,
                                (meta[kMetaNumValidSet]) * sizeof(uint32_t)));
  } else {
    dataset->indptr64 = static_cast<uint64_t *>(
        Graph::LoadDataFromFile(dataset->folder + kIndptr64File,
                                (meta[kMetaNumNode] + 1) * sizeof(uint64_t)));
    dataset->indices64 = static_cast<uint64_t *>(
        Graph::LoadDataFromFile(dataset->folder + kIndices64File,
                                meta[kMetaNumEdge] * sizeof(uint64_t)));
    dataset->train_set64 = static_cast<uint64_t *>(
        Graph::LoadDataFromFile(dataset->folder + kTrainSet64File,
                                (meta[kMetaNumTrainSet]) * sizeof(uint64_t)));
    dataset->test_set64 = static_cast<uint64_t *>(
        Graph::LoadDataFromFile(dataset->folder + kTestSet64File,
                                meta[kMetaNumTestSet] * sizeof(uint64_t)));
    dataset->valid_set64 = static_cast<uint64_t *>(
        Graph::LoadDataFromFile(dataset->folder + kValidSet64File,
                                (meta[kMetaNumValidSet]) * sizeof(uint64_t)));
  }

  dataset->feature = static_cast<float *>(Graph::LoadDataFromFile(
      dataset->folder + kFeatFile,
      meta[kMetaNumNode] * meta[kMetaFeatDim] * sizeof(float)));
  dataset->label = static_cast<uint64_t *>(Graph::LoadDataFromFile(
      dataset->folder + kLabelFile, meta[kMetaNumNode] * sizeof(uint64_t)));

  std::cout << "Loading graph with " << dataset->num_nodes << " nodes and "
            << dataset->num_edges << " edges" << std::endl;

  return dataset;
}

}  // namespace utility