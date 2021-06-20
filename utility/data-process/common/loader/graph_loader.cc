#include "graph_loader.h"

#include <fcntl.h>
#include <omp.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <iostream>

namespace utility {

GraphDataset::GraphDataset() : indptr(nullptr), indices(nullptr) {}

GraphDataset::~GraphDataset() {
  if (indptr) {
    int ret = munmap(indptr, sizeof(uint32_t) * (num_nodes + 1));
    if (ret != 0) {
      std::cout << "munmap indptr error" << std::endl;
      exit(1);
    }
  }

  if (indices) {
    int ret = munmap(indices, sizeof(uint32_t) * num_edges);
    if (ret != 0) {
      std::cout << "munmap indices error" << std::endl;
      exit(1);
    }
  }
}

const std::unordered_map<GraphCode, GraphInfo> GraphLoader::kGraphInfoMap = {
    {kComfriendster, {"com-friendster", 65608366, 1871675501}},
    {kPapers100M, {"papers100M", 111059956, 1726745828}},
    {kProducts, {"products", 2449029, 61859140}},
    {kReddit, {"reddit", 232965, 114848857}}};

const std::string GraphLoader::kIndptrFileName = "indptr.bin";
const std::string GraphLoader::kIndicesFileName = "indices.bin";

GraphLoader::GraphLoader(std::string basic_path) {
  if (basic_path.back() != '/') {
    _basic_path = basic_path + '/';
  } else {
    _basic_path = basic_path;
  }
}

std::shared_ptr<GraphDataset> GraphLoader::GetGraphDataset(GraphCode code) {
  GraphInfo graph_info = kGraphInfoMap.at(code);
  auto dataset = std::make_shared<GraphDataset>();

  dataset->folder = _basic_path + graph_info.name + '/';
  dataset->num_nodes = graph_info.num_nodes;
  dataset->num_edges = graph_info.num_edges;

  int fd;
  struct stat st;
  size_t nbytes;

  fd = open(kIndptrFileName.c_str(), O_RDONLY, 0);
  stat(kIndptrFileName.c_str(), &st);
  nbytes = st.st_size;

  if (nbytes == 0) {
    std::cout << "Reading file error: " << kIndptrFileName << std::endl;
    exit(1);
  }

  dataset->indptr =
      (uint32_t *)mmap(NULL, nbytes, PROT_READ, MAP_SHARED | MAP_FILE, fd, 0);
  mlock(dataset->indptr, nbytes);
  close(fd);

  fd = open(kIndicesFileName.c_str(), O_RDONLY, 0);
  stat(kIndicesFileName.c_str(), &st);
  nbytes = st.st_size;
  if (nbytes == 0) {
    std::cout << "Reading file error: " << kIndicesFileName << std::endl;
    exit(1);
  }

  dataset->indices =
      (uint32_t *)mmap(NULL, nbytes, PROT_READ, MAP_SHARED | MAP_FILE, fd, 0);
  mlock(dataset->indices, nbytes);
  close(fd);

  return dataset;
}

}  // namespace utility