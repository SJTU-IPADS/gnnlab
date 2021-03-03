#ifndef SAMGRAPH_CONFIG_H
#define SAMGRAPH_CONFIG_H

namespace samgraph {
namespace common {

constexpr static const std::string kMetaFile     = "meta.txt";
constexpr static const std::string kFeatFile     = "feat.bin";
constexpr static const std::string kLabelFile    = "label.bin";
constexpr static const std::string kInptrFile    = "indptr.bin";
constexpr static const std::string kIndicesFile  = "indices.bin";
constexpr static const std::string kTrainSetFile = "train_set.bin";
constexpr static const std::string kTestSetFile  = "test_set.bin";
constexpr static const std::string kValidSetFile = "valid_set.bin";

constexpr static const std::string kMetaNumNode     = "NUM_NODE";
constexpr static const std::string kMetaNumEdge     = "NUM_EDGE";
constexpr static const std::string kMetaFeatDim     = "NUM_CLASS";
constexpr static const std::string kMetaNumClass    = "NUM_CLASS";
constexpr static const std::string KMetaNumTrainSet = "NUM_TRAIN_SET";
constexpr static const std::string kMetaNumTestSet  = "NUM_TEST_SET"; 
constexpr static const std::string kMetaNumValidSet = "NUM_VALID_SET";

constexpr static const int kCudaBlockSize = 256;
constexpr static const size_t kCudaTileSize = 1024;


} // namespace common
} // namespace samgraph

#endif // SAMGRAPH_CONFIG_H
