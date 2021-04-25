#include "constant.h"

namespace samgraph {
namespace common {

const std::string Constant::kMetaFile = "meta.txt";
const std::string Constant::kFeatFile = "feat.bin";
const std::string Constant::kLabelFile = "label.bin";
const std::string Constant::kInptrFile = "indptr.bin";
const std::string Constant::kIndicesFile = "indices.bin";
const std::string Constant::kTrainSetFile = "train_set.bin";
const std::string Constant::kTestSetFile = "test_set.bin";
const std::string Constant::kValidSetFile = "valid_set.bin";

const std::string Constant::kMetaNumNode = "NUM_NODE";
const std::string Constant::kMetaNumEdge = "NUM_EDGE";
const std::string Constant::kMetaFeatDim = "FEAT_DIM";
const std::string Constant::kMetaNumClass = "NUM_CLASS";
const std::string Constant::kMetaNumTrainSet = "NUM_TRAIN_SET";
const std::string Constant::kMetaNumTestSet = "NUM_TEST_SET";
const std::string Constant::kMetaNumValidSet = "NUM_VALID_SET";

const std::string Constant::kEnvProfileLevel = "SAMGRAPH_PROFILE_LEVEL";

}  // namespace common
}  // namespace samgraph