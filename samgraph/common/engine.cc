#include "engine.h"

#include <sys/mman.h>
#include <sys/stat.h>

#include <cstdlib>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <unordered_map>

#include "common.h"
#include "constant.h"
#include "cpu/cpu_engine.h"
#include "cuda/cuda_engine.h"
#include "dist/dist_engine.h"
#include "logging.h"
#include "profiler.h"
#include "run_config.h"
#include "timer.h"

namespace samgraph {
namespace common {

Engine* Engine::_engine = nullptr;

void Engine::Create() {
  if (_engine) {
    return;
  }

  switch (RunConfig::run_arch) {
    case kArch0:
      LOG(INFO) << "Use CPU Engine (Arch " << RunConfig::run_arch << ")";
      _engine = new cpu::CPUEngine();
      break;
    case kArch1:
    case kArch2:
    case kArch3:
    case kArch4:
    case kArch7:
      LOG(INFO) << "Use GPU Engine (Arch " << RunConfig::run_arch << ")";
      _engine = new cuda::GPUEngine();
      break;
    case kArch5:
    case kArch6:
      LOG(INFO) << "Use Dist Engine (Arch " << RunConfig::run_arch << ")";
      _engine = new dist::DistEngine();
      break;
    default:
      CHECK(0);
  }
}

void Engine::LoadGraphDataset() {
  Timer t;
  // Load graph dataset from disk by mmap and copy the graph
  // topology data into the target CUDA device.
  _dataset = new Dataset;
  std::unordered_map<std::string, size_t> meta;
  std::unordered_map<std::string, Context> ctx_map = GetGraphFileCtx();

  if (_dataset_path.back() != '/') {
    _dataset_path.push_back('/');
  }

  // Parse the meta data
  std::ifstream meta_file(_dataset_path + Constant::kMetaFile);
  std::string line;
  while (std::getline(meta_file, line)) {
    std::istringstream iss(line);
    std::vector<std::string> kv{std::istream_iterator<std::string>{iss},
                                std::istream_iterator<std::string>{}};

    if (kv.size() < 2) {
      break;
    }

    meta[kv[0]] = std::stoull(kv[1]);
  }

  CHECK(meta.count(Constant::kMetaNumNode) > 0);
  CHECK(meta.count(Constant::kMetaNumEdge) > 0);
  CHECK(meta.count(Constant::kMetaFeatDim) > 0);
  CHECK(meta.count(Constant::kMetaNumClass) > 0);
  CHECK(meta.count(Constant::kMetaNumTrainSet) > 0);
  CHECK(meta.count(Constant::kMetaNumTestSet) > 0);
  CHECK(meta.count(Constant::kMetaNumValidSet) > 0);

  CHECK(ctx_map.count(Constant::kIndptrFile) > 0);
  CHECK(ctx_map.count(Constant::kIndicesFile) > 0);
  CHECK(ctx_map.count(Constant::kFeatFile) > 0);
  CHECK(ctx_map.count(Constant::kLabelFile) > 0);
  CHECK(ctx_map.count(Constant::kTrainSetFile) > 0);
  CHECK(ctx_map.count(Constant::kTestSetFile) > 0);
  CHECK(ctx_map.count(Constant::kValidSetFile) > 0);
  CHECK(ctx_map.count(Constant::kAliasTableFile) > 0);
  CHECK(ctx_map.count(Constant::kProbTableFile) > 0);
  CHECK(ctx_map.count(Constant::kInDegreeFile) > 0);
  CHECK(ctx_map.count(Constant::kOutDegreeFile) > 0);
  CHECK(ctx_map.count(Constant::kCacheByDegreeFile) > 0);
  CHECK(ctx_map.count(Constant::kCacheByHeuristicFile) > 0);
  CHECK(ctx_map.count(Constant::kCacheByDegreeHopFile) > 0);
  CHECK(ctx_map.count(Constant::kCacheByFakeOptimalFile) > 0);
  CHECK(ctx_map.count(Constant::kCacheByRandomFile) > 0);

  _dataset->num_node = meta[Constant::kMetaNumNode];
  _dataset->num_edge = meta[Constant::kMetaNumEdge];
  _dataset->num_class = meta[Constant::kMetaNumClass];

  _dataset->indptr =
      Tensor::FromMmap(_dataset_path + Constant::kIndptrFile, DataType::kI32,
                       {meta[Constant::kMetaNumNode] + 1},
                       ctx_map[Constant::kIndptrFile], "dataset.indptr");
  _dataset->indices =
      Tensor::FromMmap(_dataset_path + Constant::kIndicesFile, DataType::kI32,
                       {meta[Constant::kMetaNumEdge]},
                       ctx_map[Constant::kIndicesFile], "dataset.indices");

  if (FileExist(_dataset_path + Constant::kFeatFile) && RunConfig::option_empty_feat == 0) {
    _dataset->feat = Tensor::FromMmap(
        _dataset_path + Constant::kFeatFile, DataType::kF32,
        {meta[Constant::kMetaNumNode], meta[Constant::kMetaFeatDim]},
        ctx_map[Constant::kFeatFile], "dataset.feat");
  } else {
    if (RunConfig::option_empty_feat != 0) {
      _dataset->feat = Tensor::EmptyNoScale(
          DataType::kF32,
          {1ull << RunConfig::option_empty_feat, meta[Constant::kMetaFeatDim]},
          ctx_map[Constant::kFeatFile], "dataset.feat");
    } else {
      _dataset->feat = Tensor::EmptyNoScale(
          DataType::kF32,
          {meta[Constant::kMetaNumNode], meta[Constant::kMetaFeatDim]},
          ctx_map[Constant::kFeatFile], "dataset.feat");
    }
  }

  if (FileExist(_dataset_path + Constant::kLabelFile)) {
    _dataset->label =
        Tensor::FromMmap(_dataset_path + Constant::kLabelFile, DataType::kI64,
                         {meta[Constant::kMetaNumNode]},
                         ctx_map[Constant::kLabelFile], "dataset.label");
  } else {
    _dataset->label =
        Tensor::EmptyNoScale(DataType::kI64, {meta[Constant::kMetaNumNode]},
                             ctx_map[Constant::kLabelFile], "dataset.label");
  }

  _dataset->train_set =
      Tensor::FromMmap(_dataset_path + Constant::kTrainSetFile, DataType::kI32,
                       {meta[Constant::kMetaNumTrainSet]},
                       ctx_map[Constant::kTrainSetFile], "dataset.train_set");
  _dataset->test_set =
      Tensor::FromMmap(_dataset_path + Constant::kTestSetFile, DataType::kI32,
                       {meta[Constant::kMetaNumTestSet]},
                       ctx_map[Constant::kTestSetFile], "dataset.test_set");
  _dataset->valid_set =
      Tensor::FromMmap(_dataset_path + Constant::kValidSetFile, DataType::kI32,
                       {meta[Constant::kMetaNumValidSet]},
                       ctx_map[Constant::kValidSetFile], "dataset.valid_set");

  if (RunConfig::sample_type == kWeightedKHop || RunConfig::sample_type == kWeightedKHopHashDedup) {
    _dataset->prob_table = Tensor::FromMmap(
        _dataset_path + Constant::kProbTableFile, DataType::kF32,
        {meta[Constant::kMetaNumEdge]}, ctx_map[Constant::kProbTableFile],
        "dataset.prob_table");

    _dataset->alias_table = Tensor::FromMmap(
        _dataset_path + Constant::kAliasTableFile, DataType::kI32,
        {meta[Constant::kMetaNumEdge]}, ctx_map[Constant::kAliasTableFile],
        "dataset.alias_table");
    _dataset->prob_prefix_table = Tensor::Null();
  } else if (RunConfig::sample_type == kWeightedKHopPrefix){
    _dataset->prob_table = Tensor::Null();
    _dataset->alias_table = Tensor::Null();
    _dataset->prob_prefix_table = Tensor::FromMmap(
        _dataset_path + Constant::kProbPrefixTableFile, DataType::kF32,
        {meta[Constant::kMetaNumEdge]}, ctx_map[Constant::kProbTableFile],
        "dataset.prob_prefix_table");
  } else {
    _dataset->prob_table = Tensor::Null();
    _dataset->alias_table = Tensor::Null();
    _dataset->prob_prefix_table = Tensor::Null();
  }

  if (RunConfig::option_log_node_access) {
    _dataset->in_degrees = Tensor::FromMmap(
        _dataset_path + Constant::kInDegreeFile, DataType::kI32,
        {meta[Constant::kMetaNumNode]}, ctx_map[Constant::kInDegreeFile],
        "dataset.in_degrees");
    _dataset->out_degrees = Tensor::FromMmap(
        _dataset_path + Constant::kOutDegreeFile, DataType::kI32,
        {meta[Constant::kMetaNumNode]}, ctx_map[Constant::kOutDegreeFile],
        "dataset.out_degrees");
  }

  if (RunConfig::UseGPUCache()) {
    switch (RunConfig::cache_policy) {
      case kCacheByDegree:
        _dataset->ranking_nodes = Tensor::FromMmap(
            _dataset_path + Constant::kCacheByDegreeFile, DataType::kI32,
            {meta[Constant::kMetaNumNode]},
            ctx_map[Constant::kCacheByDegreeFile], "dataset.ranking_nodes");
        break;
      case kCacheByHeuristic:
        _dataset->ranking_nodes = Tensor::FromMmap(
            _dataset_path + Constant::kCacheByHeuristicFile, DataType::kI32,
            {meta[Constant::kMetaNumNode]},
            ctx_map[Constant::kCacheByHeuristicFile], "dataset.ranking_nodes");
        break;
      case kCacheByPreSample:
      case kCacheByPreSampleStatic:
        break;
      case kCacheByDegreeHop:
        _dataset->ranking_nodes = Tensor::FromMmap(
            _dataset_path + Constant::kCacheByDegreeHopFile, DataType::kI32,
            {meta[Constant::kMetaNumNode]},
            ctx_map[Constant::kCacheByDegreeHopFile], "dataset.ranking_nodes");
        break;
      case kCacheByFakeOptimal:
        _dataset->ranking_nodes = Tensor::FromMmap(
            _dataset_path + Constant::kCacheByFakeOptimalFile, DataType::kI32,
            {meta[Constant::kMetaNumNode]},
            ctx_map[Constant::kCacheByFakeOptimalFile], "dataset.ranking_nodes");
        break;
      case kCacheByRandom:
        _dataset->ranking_nodes = Tensor::FromMmap(
            _dataset_path + Constant::kCacheByRandomFile, DataType::kI32,
            {meta[Constant::kMetaNumNode]},
            ctx_map[Constant::kCacheByRandomFile], "dataset.ranking_nodes");
        break;
      case kDynamicCache:
        break;
      default:
        CHECK(0);
    }
  }

  double loading_time = t.Passed();
  LOG(INFO) << "SamGraph loaded dataset(" << _dataset_path << ") successfully ("
            << loading_time << " secs)";
  LOG(DEBUG) << "dataset(" << _dataset_path << ") has "
             << _dataset->num_node << " nodes, "
             << _dataset->num_edge << " edges ";
}

bool Engine::IsAllThreadFinish(int total_thread_num) {
  int k = _joined_thread_cnt.fetch_add(0);
  return (k == total_thread_num);
};

void Engine::ForwardBarrier() {
  outer_counter++;
}
void Engine::ForwardInnerBarrier() {
  inner_counter++;
}

}  // namespace common
}  // namespace samgraph
