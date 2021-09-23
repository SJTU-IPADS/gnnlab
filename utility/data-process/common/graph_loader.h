#ifndef UTILITY_COMMON_GRAPH_LOADER_H
#define UTILITY_COMMON_GRAPH_LOADER_H

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace utility {

class Graph {
 public:
  std::string folder;
  size_t num_nodes;
  size_t num_edges;
  size_t num_train_set;
  size_t num_test_set;
  size_t num_valid_set;

  uint32_t *indptr;
  uint32_t *indices;
  uint32_t *train_set;
  uint32_t *valid_set;
  uint32_t *test_set;

  size_t feat_dim;
  float *feature;
  uint64_t *label;

  uint64_t *indptr64;
  uint64_t *indices64;
  uint64_t *train_set64;
  uint64_t *valid_set64;
  uint64_t *test_set64;

  Graph();
  ~Graph();

  static void *LoadDataFromFile(std::string file, size_t expected_size);
};

using GraphPtr = std::shared_ptr<Graph>;

class DegreeInfo {
 public:
  std::vector<uint32_t> in_degrees;
  std::vector<uint32_t> out_degrees;
  static std::shared_ptr<DegreeInfo> GetDegrees(GraphPtr &graph);
};

class GraphLoader {
 public:
  GraphLoader(std::string root);
  GraphPtr GetGraphDataset(std::string graph, bool is64type = false);

  static const std::string kMetaFile;
  static const std::string kFeatFile;
  static const std::string kLabelFile;
  static const std::string kIndptrFile;
  static const std::string kIndicesFile;
  static const std::string kTrainSetFile;
  static const std::string kTestSetFile;
  static const std::string kValidSetFile;

  static const std::string kIndptr64File;
  static const std::string kIndices64File;
  static const std::string kTrainSet64File;
  static const std::string kTestSet64File;
  static const std::string kValidSet64File;

  static const std::string kMetaNumNode;
  static const std::string kMetaNumEdge;
  static const std::string kMetaFeatDim;
  static const std::string kMetaNumClass;
  static const std::string kMetaNumTrainSet;
  static const std::string kMetaNumTestSet;
  static const std::string kMetaNumValidSet;

 private:
  std::string _root;
};

}  // namespace utility

#endif  // UTILITY_COMMON_GRAPH_LOADER_H