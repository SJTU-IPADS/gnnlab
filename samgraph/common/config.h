#ifndef SAMGRAPH_CONFIG_H
#define SAMGRAPH_CONFIG_H

#include <unordered_map>
#include <cstdint>
#include <limits>

#include "common.h"
#include "types.h"

namespace samgraph {
namespace common {

class Config {
 public:
  static const std::string kMetaFile;
  static const std::string kFeatFile;
  static const std::string kLabelFile;
  static const std::string kInptrFile;
  static const std::string kIndicesFile;
  static const std::string kTrainSetFile;
  static const std::string kTestSetFile;
  static const std::string kValidSetFile;

  static const std::string kMetaNumNode;
  static const std::string kMetaNumEdge;
  static const std::string kMetaFeatDim;
  static const std::string kMetaNumClass;
  static const std::string kMetaNumTrainSet;
  static const std::string kMetaNumTestSet; 
  static const std::string kMetaNumValidSet;

  static constexpr int kCudaBlockSize = 256;
  static constexpr size_t kCudaTileSize = 1024;

  static const std::unordered_map<int, size_t> kQueueThreshold;
  static constexpr size_t kGraphPoolThreshold = 5;

  static constexpr IdType kEmptyKey = std::numeric_limits<IdType>::max();

  static constexpr uint64_t kBatchMask = ~0xffff;
  static constexpr uint64_t kGraphMask = 0xffff;

  static constexpr size_t kBufferSize = 64;
  static constexpr size_t kGigabytes = 1 * 1024 * 1024 * 1024;
  static constexpr size_t kMegabytes = 1 * 1024 * 1024;
  static constexpr size_t kKilobytes = 1 * 1024;
};

} // namespace common
} // namespace samgraph

#endif // SAMGRAPH_CONFIG_H