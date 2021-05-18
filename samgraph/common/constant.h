#ifndef SAMGRAPH_CONFIG_H
#define SAMGRAPH_CONFIG_H

#include <cstdint>
#include <limits>
#include <unordered_map>

#include "common.h"

namespace samgraph {
namespace common {

class Constant {
 public:
  static const std::string kMetaFile;
  static const std::string kFeatFile;
  static const std::string kLabelFile;
  static const std::string kInptrFile;
  static const std::string kIndicesFile;
  static const std::string kTrainSetFile;
  static const std::string kTestSetFile;
  static const std::string kValidSetFile;

  static const std::string kInDegreeFile;
  static const std::string kOutDegreeFile;

  static const std::string kMetaNumNode;
  static const std::string kMetaNumEdge;
  static const std::string kMetaFeatDim;
  static const std::string kMetaNumClass;
  static const std::string kMetaNumTrainSet;
  static const std::string kMetaNumTestSet;
  static const std::string kMetaNumValidSet;

  static constexpr int kCudaBlockSize = 256;
  static constexpr size_t kCudaTileSize = 1024;

  static constexpr IdType kEmptyKey = std::numeric_limits<IdType>::max();

  static constexpr size_t kBufferSize = 64;
  static constexpr size_t kGigabytes = 1 * 1024 * 1024 * 1024;
  static constexpr size_t kMegabytes = 1 * 1024 * 1024;
  static constexpr size_t kKilobytes = 1 * 1024;

  static constexpr size_t kAllocScale = 2;
  static constexpr size_t kAllocNoScale = 1;

  static const std::string kEnvProfileLevel;
  static const std::string kEnvProfileCuda;
  static const std::string kEnvLogNodeAccess;

  static const std::string kNodeAccessLogFile;
  static const std::string kNodeAccessFrequencyFile;
  static const std::string kNodeAccessFileSuffix;
};

}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_CONFIG_H
