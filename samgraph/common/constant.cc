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

#include "constant.h"
#include <cstdlib>

namespace samgraph {
namespace common {

const std::string Constant::kMetaFile = "meta.txt";
const std::string Constant::kFeatFile = "feat.bin";
const std::string Constant::kLabelFile = "label.bin";
const std::string Constant::kIndptrFile = "indptr.bin";
const std::string Constant::kIndicesFile = "indices.bin";
const std::string Constant::kTrainSetFile = "train_set.bin";
const std::string Constant::kTestSetFile = "test_set.bin";
const std::string Constant::kValidSetFile = "valid_set.bin";

const std::string Constant::kLinkTrainSetFile = "link_train_set.bin";
const std::string Constant::kLinkTestSetFile = "link_test_set.bin";
const std::string Constant::kLinkValidSetFile = "link_valid_set.bin";

const std::string Constant::kProbTableFile = "prob_table.bin";
const std::string Constant::kAliasTableFile = "alias_table.bin";
const std::string Constant::kProbPrefixTableFile = "prob_prefix_table.bin";

const std::string Constant::kInDegreeFile = "in_degrees.bin";
const std::string Constant::kOutDegreeFile = "out_degrees.bin";
const std::string Constant::kCacheByDegreeFile = "cache_by_degree.bin";
const std::string Constant::kCacheByHeuristicFile = "cache_by_heuristic.bin";
const std::string Constant::kCacheByDegreeHopFile = "cache_by_degree_hop.bin";
const std::string Constant::kCacheByFakeOptimalFile = "cache_by_fake_optimal.bin";
const std::string Constant::kCacheByRandomFile = "cache_by_random.bin";

const std::string Constant::kMetaNumNode = "NUM_NODE";
const std::string Constant::kMetaNumEdge = "NUM_EDGE";
const std::string Constant::kMetaFeatDim = "FEAT_DIM";
const std::string Constant::kMetaFeatDataType = "FEAT_DATA_TYPE";
const std::string Constant::kMetaNumClass = "NUM_CLASS";
const std::string Constant::kMetaNumTrainSet = "NUM_TRAIN_SET";
const std::string Constant::kMetaNumTestSet = "NUM_TEST_SET";
const std::string Constant::kMetaNumValidSet = "NUM_VALID_SET";

const std::string Constant::kMetaNumLinkTrainSet = "NUM_LINK_TRAIN_SET";
const std::string Constant::kMetaNumLinkTestSet = "NUM_LINK_TEST_SET";
const std::string Constant::kMetaNumLinkValidSet = "NUM_LINK_VALID_SET";

const std::string Constant::kOMPNumThreads = "SAMGRAPH_OMP_NUM_THREADS";
const std::string Constant::kEnvProfileLevel = "SAMGRAPH_PROFILE_LEVEL";
const std::string Constant::kEnvProfileCuda = "SAMGRAPH_PROFILE_CUDA";
const std::string Constant::kEnvLogNodeAccess = "SAMGRAPH_LOG_NODE_ACCESS";
const std::string Constant::kEnvLogNodeAccessSimple = "SAMGRAPH_LOG_NODE_ACCESS_SIMPLE";
const std::string Constant::kEnvSanityCheck = "SAMGRAPH_SANITY_CHECK";
const std::string Constant::kEnvDumpTrace = "SAMGRAPH_DUMP_TRACE";
const std::string Constant::kEnvEmptyFeat = "SAMGRAPH_EMPTY_FEAT";
const std::string Constant::kEnvSamBackCudaLaunchBlocking = "SAMBACK_CUDA_LAUNCH_BLOCKING";
const std::string Constant::kEnvTrainSetPart = "SAMGRAPH_TRAIN_SET_PART";
const std::string Constant::kEnvFakeFeatDim = "SAMGRAPH_FAKE_FEAT_DIM";

const std::string Constant::kNodeAccessLogFile = "node_access";
const std::string Constant::kNodeAccessFrequencyFile = "node_access_frequency";
const std::string Constant::kNodeAccessOptimalCacheHitFile = "node_access_optimal_cache_hit";
const std::string Constant::kNodeAccessOptimalCacheBinFile = "node_access_optimal_cache_bin";
const std::string Constant::kNodeAccessOptimalCacheFreqBinFile = "node_access_optimal_cache_freq_bin";
const std::string Constant::kNodeAccessSimilarityFile = "node_access_similarity";
const std::string Constant::kNodeAccessPreSampleSimFile = "node_access_presample";
const std::string Constant::kNodeAccessFileSuffix = ".txt";

const std::string Constant::kCollCacheBuilderShmName = std::string("coll_cache_shm_") + std::string(std::getenv("USER"));
const std::string Constant::kCollCachePlacementShmName = std::string("coll_cache_block_placement_") + std::string(std::getenv("USER"));
const std::string Constant::kCollCacheAccessShmName = std::string("coll_cache_block_access_from_") + std::string(std::getenv("USER"));

const std::string Constant::kProfilerValShmName    = std::string("coll_cache_val_shm_") + std::string(std::getenv("USER"));
const std::string Constant::kProfilerBitmapShmName = std::string("coll_cache_bitmap_shm_") + std::string(std::getenv("USER"));

}  // namespace common
}  // namespace samgraph
