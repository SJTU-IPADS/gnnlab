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

#ifndef SAMGRAPH_CONSTANT_H
#define SAMGRAPH_CONSTANT_H

#include <cstdint>
#include <limits>
#include <unordered_map>

namespace samgraph {
namespace common {

using IdType = unsigned int;
using Id64Type = unsigned long long int;
static_assert(sizeof(Id64Type) == 8, "long long is not 8 bytes!");
static_assert(sizeof(short) == 2, "short is not 2 bytes!");

class Constant {
 public:
  static const std::string kMetaFile;
  static const std::string kFeatFile;
  static const std::string kLabelFile;
  static const std::string kIndptrFile;
  static const std::string kIndicesFile;
  static const std::string kTrainSetFile;
  static const std::string kTestSetFile;
  static const std::string kValidSetFile;

  static const std::string kLinkTrainSetFile;
  static const std::string kLinkTestSetFile;
  static const std::string kLinkValidSetFile;

  static const std::string kProbTableFile;
  static const std::string kAliasTableFile;
  static const std::string kProbPrefixTableFile;

  static const std::string kInDegreeFile;
  static const std::string kOutDegreeFile;

  static const std::string kCacheByDegreeFile;
  static const std::string kCacheByHeuristicFile;
  static const std::string kCacheByDegreeHopFile;
  static const std::string kCacheByFakeOptimalFile;
  static const std::string kCacheByRandomFile;

  static const std::string kMetaNumNode;
  static const std::string kMetaNumEdge;
  static const std::string kMetaFeatDim;
  static const std::string kMetaFeatDataType;
  static const std::string kMetaNumClass;
  static const std::string kMetaNumTrainSet;
  static const std::string kMetaNumTestSet;
  static const std::string kMetaNumValidSet;

  static const std::string kMetaNumLinkTrainSet;
  static const std::string kMetaNumLinkTestSet;
  static const std::string kMetaNumLinkValidSet;

  static constexpr size_t kCudaBlockSize = 256;
  static constexpr size_t kCudaTileSize = 1024;

  // In nextdoor, this value is set to 5 * 1024 * 1024,
  // but we find that 512 * 1024 is the best value in V100
  static constexpr size_t kKHop1MaxThreads = 512 * 1024;
  static constexpr size_t kWeightedKHopMaxThreads = 512 * 1024;
  static constexpr size_t kRandomWalkMaxThreads = 512 * 1024;

  static constexpr IdType kEmptyKey = std::numeric_limits<IdType>::max();
  static constexpr Id64Type kEmptyLabel = std::numeric_limits<Id64Type>::max();

  static constexpr size_t kBufferSize = 64;
  static constexpr size_t kGigabytes = 1 * 1024 * 1024 * 1024;
  static constexpr size_t kMegabytes = 1 * 1024 * 1024;
  static constexpr size_t kKilobytes = 1 * 1024;

  static constexpr double kAllocScale = 1.25f;
  static constexpr double kAllocNoScale = 1.0f;

  static const std::string kOMPNumThreads;
  static const std::string kEnvProfileLevel;
  static const std::string kEnvProfileCuda;
  static const std::string kEnvLogNodeAccess;
  static const std::string kEnvLogNodeAccessSimple;
  static const std::string kEnvSanityCheck;
  static const std::string kEnvDumpTrace;
  static const std::string kEnvEmptyFeat;
  static const std::string kEnvSamBackCudaLaunchBlocking;
  static const std::string kEnvTrainSetPart;
  static const std::string kEnvFakeFeatDim;

  static const std::string kNodeAccessLogFile;
  static const std::string kNodeAccessFrequencyFile;
  static const std::string kNodeAccessOptimalCacheHitFile;
  static const std::string kNodeAccessOptimalCacheBinFile;
  static const std::string kNodeAccessOptimalCacheFreqBinFile;
  static const std::string kNodeAccessSimilarityFile;
  static const std::string kNodeAccessPreSampleSimFile;
  static const std::string kNodeAccessFileSuffix;
};

}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_CONSTANT_H
