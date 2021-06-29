#include <fstream>
#include <string>
#include <vector>

#include "common/graph_loader.h"
#include "common/options.h"
#include "common/utils.h"

void toFile(const std::vector<uint64_t> &data, const std::string filename) {
  std::ofstream ofs(filename, std::ofstream::out | std::ofstream::binary |
                                  std::ofstream::trunc);
  ofs.write((const char *)data.data(), data.size() * sizeof(uint64_t));

  ofs.close();
}

void to64(utility::GraphPtr graph) {
  std::string folder = graph->folder;
  uint32_t *indptr = graph->indptr;
  uint32_t *indices = graph->indices;
  uint32_t *train_set = graph->train_set;
  uint32_t *valid_set = graph->valid_set;
  uint32_t *test_set = graph->test_set;

  size_t num_nodes = graph->num_nodes;
  size_t num_edges = graph->num_edges;
  size_t num_train_set = graph->num_train_set;
  size_t num_test_set = graph->num_test_set;
  size_t num_valid_set = graph->num_valid_set;

  std::vector<uint64_t> indptr64(num_nodes + 1);
  std::vector<uint64_t> indices64(num_edges);
  std::vector<uint64_t> train_set64(num_train_set);
  std::vector<uint64_t> test_set64(num_test_set);
  std::vector<uint64_t> valid_set64(num_valid_set);

  // #pragma omp parallel for
  //   for (size_t i = 0; i < (num_nodes + 1); i++) {
  //     indptr64[i] = indptr[i];
  //   }

  // #pragma omp parallel for
  //   for (size_t i = 0; i < num_edges; i++) {
  //     indices64[i] = indices[i];
  //   }

#pragma omp parallel for
  for (size_t i = 0; i < num_train_set; i++) {
    train_set64[i] = train_set[i];
  }

#pragma omp parallel for
  for (size_t i = 0; i < num_test_set; i++) {
    test_set64[i] = test_set[i];
  }

#pragma omp parallel for
  for (size_t i = 0; i < num_valid_set; i++) {
    valid_set64[i] = valid_set[i];
  }

  //   toFile(indptr64, folder + "indptr64.bin");
  //   toFile(indices64, folder + "indices64.bin");
  toFile(train_set64, folder + "train_set64.bin");
  toFile(test_set64, folder + "test_set64.bin");
  toFile(valid_set64, folder + "valid_set64.bin");
}

int main(int argc, char *argv[]) {
  utility::Options::InitOptions("Graph property");
  OPTIONS_PARSE(argc, argv);

  utility::GraphLoader graph_loader(utility::Options::root);
  auto graph = graph_loader.GetGraphDataset(utility::Options::graph);

  to64(graph);
}
