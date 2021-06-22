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

const std::string GraphLoader::kMetaNumNode = "NUM_NODE";
const std::string GraphLoader::kMetaNumEdge = "NUM_EDGE";
const std::string GraphLoader::kMetaFeatDim = "FEAT_DIM";
const std::string GraphLoader::kMetaNumClass = "NUM_CLASS";
const std::string GraphLoader::kMetaNumTrainSet = "NUM_TRAIN_SET";
const std::string GraphLoader::kMetaNumTestSet = "NUM_TEST_SET";
const std::string GraphLoader::kMetaNumValidSet = "NUM_VALID_SET";

Graph::Graph() : indptr(nullptr), indices(nullptr) {}

Graph::~Graph() {
  if (indptr) {
    int ret = munmap(indptr, sizeof(uint32_t) * (num_nodes + 1));
    Check(ret == 0, "munmap indptr error");
  }

  if (indices) {
    int ret = munmap(indices, sizeof(uint32_t) * num_edges);
    Check(ret == 0, "munmap indices error");
  }
}

void *Graph::LoadDataFromFile(std::string file, const size_t expected_nbytes) {
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

GraphLoader::GraphLoader(std::string root) {
  if (root.back() != '/') {
    _root = root + '/';
  } else {
    _root = root;
  }
}

GraphPtr GraphLoader::GetGraphDataset(std::string graph) {
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

    meta[kv[0]] = std::stoull(kv[1]);
  }

  Check(meta.count(kMetaNumNode) > 0, kMetaNumNode + " not exit");
  Check(meta.count(kMetaNumEdge) > 0, kMetaNumEdge + " not exit");
  Check(meta.count(kMetaFeatDim) > 0, kMetaFeatDim + " not exit");
  Check(meta.count(kMetaNumClass) > 0, kMetaNumClass + " not exit");
  Check(meta.count(kMetaNumTrainSet) > 0, kMetaNumTrainSet + " not exit");
  Check(meta.count(kMetaNumTestSet) > 0, kMetaNumTestSet + " not exit");
  Check(meta.count(kMetaNumValidSet) > 0, kMetaNumValidSet + " not exit");

  dataset->num_nodes = meta[kMetaNumNode];
  dataset->num_edges = meta[kMetaNumEdge];

  dataset->indptr = static_cast<uint32_t *>(
      Graph::LoadDataFromFile(dataset->folder + kIndptrFile,
                              (meta[kMetaNumNode] + 1) * sizeof(uint32_t)));
  dataset->indices = static_cast<uint32_t *>(Graph::LoadDataFromFile(
      dataset->folder + kIndicesFile, meta[kMetaNumEdge] * sizeof(uint32_t)));

  std::cout << "Loading graph with " << dataset->num_nodes << " nodes and "
            << dataset->num_edges << " edges" << std::endl;

  return dataset;
}

}  // namespace utility