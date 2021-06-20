#ifndef UTILITY_COMMON_GRAPH_LOADER_H
#define UTILITY_COMMON_GRAPH_LOADER_H

#include <memory>
#include <string>
#include <unordered_map>

namespace utility {

enum GraphCode { kComfriendster = 0, kPapers100M, kProducts, kReddit };

struct GraphInfo {
  std::string name;
  size_t num_nodes;
  size_t num_edges;
};

struct GraphDataset {
  std::string folder;
  size_t num_nodes;
  size_t num_edges;

  uint32_t *indptr;
  uint32_t *indices;

  GraphDataset();
  ~GraphDataset();
}

class GraphLoader {
 public:
  GraphLoader(std::string basic_path);
  std::shared_ptr<GraphDataset> GetGraphDataset(GraphCode graph_code);

 private:
  std::string _basic_path;
  static const std::unordered_map<GraphCode, GraphInfo> kGraphInfoMap;
  static const std::string kIndptrFileName;
  static const std::string kIndicesFileName;
};

}  // namespace utility

#endif  // UTILITY_COMMON_GRAPH_LOADER_H