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

#include "engine.h"

#include <sys/mman.h>
#include <sys/stat.h>

#include <cstdlib>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <unordered_map>
#include <parallel/algorithm>
#include <parallel/numeric>

#include "common.h"
#include "constant.h"
#include "cpu/cpu_engine.h"
#include "cuda/cuda_engine.h"
#include "cuda/um_pre_sampler.h"
#include "dist/dist_engine.h"
#include "logging.h"
#include "profiler.h"
#include "run_config.h"
#include "timer.h"
#include "device.h"
#include "utils.h"

namespace samgraph {
namespace common {

namespace {

void shuffle(uint32_t * data, size_t num_data, const size_t* shuffle_range=nullptr, uint64_t seed= 0x1234567890abcdef) {
  auto g = std::default_random_engine(seed);
  if (shuffle_range == nullptr) {
    shuffle_range = & num_data;
  }
  for (size_t i = 0; i < *shuffle_range; i++) {
    std::uniform_int_distribution<size_t> d(i, num_data - 1);
    size_t candidate = d(g);
    std::swap(data[i], data[candidate]);
  }
}

};

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
    case kArch9:
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

  // default feature type is 32-bit float.
  // legacy dataset doesnot have this meta
  DataType feat_data_type = kF32;

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

    if (kv[0] == Constant::kMetaFeatDataType) {
      feat_data_type = DataTypeParseName(kv[1]);
    } else {
      meta[kv[0]] = std::stoull(kv[1]);
    }
  }

  CHECK(meta.count(Constant::kMetaNumNode) > 0);
  CHECK(meta.count(Constant::kMetaNumEdge) > 0);
  CHECK(meta.count(Constant::kMetaFeatDim) > 0);
  CHECK(meta.count(Constant::kMetaNumClass) > 0);

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

  if (!RunConfig::unsupervised_sample) {
    CHECK(meta.count(Constant::kMetaNumTrainSet) > 0);
    CHECK(meta.count(Constant::kMetaNumTestSet) > 0);
    CHECK(meta.count(Constant::kMetaNumValidSet) > 0);
  } else {
    if (meta.count(Constant::kMetaNumLinkTrainSet) > 0) {
      CHECK(meta.count(Constant::kMetaNumLinkTestSet) > 0);
      CHECK(meta.count(Constant::kMetaNumLinkValidSet) > 0);
    } else {
      meta[Constant::kMetaNumLinkTrainSet] = 0.8 * meta[Constant::kMetaNumEdge];
      meta[Constant::kMetaNumLinkTestSet]  = 0.1 * meta[Constant::kMetaNumEdge];
      meta[Constant::kMetaNumLinkValidSet] = meta[Constant::kMetaNumEdge] - meta[Constant::kMetaNumLinkTrainSet] - meta[Constant::kMetaNumLinkTestSet];
      if (meta[Constant::kMetaNumLinkTrainSet] / RunConfig::batch_size > RunConfig::step_max_boundary) {
        meta[Constant::kMetaNumLinkTrainSet] = RoundUp(RunConfig::step_max_boundary, LCM(RunConfig::num_sample_worker, RunConfig::num_train_worker)) * RunConfig::batch_size;
        meta[Constant::kMetaNumLinkTrainSet] = Min(meta[Constant::kMetaNumEdge] - meta[Constant::kMetaNumLinkTestSet] - meta[Constant::kMetaNumLinkValidSet], meta[Constant::kMetaNumLinkTrainSet]);
      }
    }
    meta[Constant::kMetaNumTrainSet] = meta[Constant::kMetaNumLinkTrainSet];
    meta[Constant::kMetaNumTestSet]  = meta[Constant::kMetaNumLinkTestSet];
    meta[Constant::kMetaNumValidSet] = meta[Constant::kMetaNumLinkValidSet];
  }

  _dataset->num_node = meta[Constant::kMetaNumNode];
  _dataset->num_edge = meta[Constant::kMetaNumEdge];
  _dataset->num_class = meta[Constant::kMetaNumClass];

  if (ctx_map[Constant::kIndptrFile].device_type != DeviceType::kGPU_UM) {
    _dataset->indptr =
        Tensor::FromMmap(_dataset_path + Constant::kIndptrFile, DataType::kI32,
                        {meta[Constant::kMetaNumNode] + 1},
                        ctx_map[Constant::kIndptrFile], "dataset.indptr");
  } else {
    _dataset->indptr = 
        Tensor::UMFromMmap(_dataset_path + Constant::kIndptrFile, DataType::kI32,
                          {meta[Constant::kMetaNumNode] + 1},
                          RunConfig::unified_memory_ctxes, "dataset.indptr");
  }
  if (ctx_map[Constant::kIndicesFile].device_type != DeviceType::kGPU_UM) {
    _dataset->indices =
        Tensor::FromMmap(_dataset_path + Constant::kIndicesFile, DataType::kI32,
                        {meta[Constant::kMetaNumEdge]},
                        ctx_map[Constant::kIndicesFile], "dataset.indices");
  } else {
    _dataset->indices =
        Tensor::UMFromMmap(_dataset_path + Constant::kIndicesFile, DataType::kI32,
                          {meta[Constant::kMetaNumEdge]},
                          RunConfig::unified_memory_ctxes, "dataset.indices");
  }

  if (RunConfig::option_fake_feat_dim != 0) {
    meta[Constant::kMetaFeatDim] = RunConfig::option_fake_feat_dim;
    _dataset->feat = Tensor::EmptyNoScale(feat_data_type,
                                          {meta[Constant::kMetaNumNode], meta[Constant::kMetaFeatDim]},
                                          ctx_map[Constant::kFeatFile], "dataset.feat");
  } else if (FileExist(_dataset_path + Constant::kFeatFile) && RunConfig::option_empty_feat == 0) {
    _dataset->feat = Tensor::FromMmap(
        _dataset_path + Constant::kFeatFile, feat_data_type,
        {meta[Constant::kMetaNumNode], meta[Constant::kMetaFeatDim]},
        ctx_map[Constant::kFeatFile], "dataset.feat");
  } else {
    if (RunConfig::option_empty_feat != 0) {
      _dataset->feat = Tensor::EmptyNoScale(
          feat_data_type,
          {1ull << RunConfig::option_empty_feat, meta[Constant::kMetaFeatDim]},
          ctx_map[Constant::kFeatFile], "dataset.feat");
    } else {
      _dataset->feat = Tensor::EmptyNoScale(
          feat_data_type,
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

  bool slice_train_should_shuffle = true;
  if (RunConfig::unsupervised_sample == false) {
    _dataset->train_set = Tensor::FromMmap(_dataset_path + Constant::kTrainSetFile, 
        DataType::kI32, {meta[Constant::kMetaNumTrainSet]},
        ctx_map[Constant::kTrainSetFile], "dataset.train_set");
    _dataset->test_set = Tensor::FromMmap(_dataset_path + Constant::kTestSetFile, 
        DataType::kI32, {meta[Constant::kMetaNumTestSet]},
        ctx_map[Constant::kTestSetFile], "dataset.test_set");
    _dataset->valid_set = Tensor::FromMmap(_dataset_path + Constant::kValidSetFile, 
        DataType::kI32, {meta[Constant::kMetaNumValidSet]},
        ctx_map[Constant::kValidSetFile], "dataset.valid_set");
  } else if (FileExist(_dataset_path + Constant::kLinkTrainSetFile)) {
    _dataset->train_set = Tensor::FromMmap(_dataset_path + Constant::kLinkTrainSetFile, 
        DataType::kI32, {meta[Constant::kMetaNumTrainSet]},
        ctx_map[Constant::kTrainSetFile], "dataset.train_set");
    // _dataset->test_set = Tensor::FromMmap(_dataset_path + Constant::kLinkTestSetFile, 
    //     DataType::kI32, {meta[Constant::kMetaNumTestSet]},
    //     ctx_map[Constant::kTestSetFile], "dataset.test_set");
    // _dataset->valid_set = Tensor::FromMmap(_dataset_path + Constant::kLinkValidSetFile, 
    //     DataType::kI32, {meta[Constant::kMetaNumValidSet]},
    //     ctx_map[Constant::kValidSetFile], "dataset.valid_set");
  } else {
    // no train set file. randomly generate from all edge
    auto cpu_ctx = CPU_CLIB();
    auto full_eid = Tensor::EmptyNoScale(kI32, {_dataset->num_edge}, cpu_ctx, "full_eid");
    cpu::ArrangeArray(full_eid->Ptr<IdType>(), _dataset->num_edge);
    // size_t efficient_shuffle_range = meta[Constant::kMetaNumTrainSet] + meta[Constant::kMetaNumTestSet] + meta[Constant::kMetaNumValidSet];
    size_t efficient_shuffle_range = meta[Constant::kMetaNumTrainSet];
    shuffle(full_eid->Ptr<IdType>(), _dataset->num_edge, &efficient_shuffle_range);
    slice_train_should_shuffle = false;
    _dataset->train_set = Tensor::CopyBlob(full_eid->Ptr<IdType>(),
        DataType::kI32, {meta[Constant::kMetaNumTrainSet]}, cpu_ctx, ctx_map[Constant::kTrainSetFile], "dataset.train_set");
    // _dataset->test_set  = Tensor::CopyBlob(full_eid->Ptr<IdType>() + meta[Constant::kMetaNumTrainSet],
    //     DataType::kI32, {meta[Constant::kMetaNumTestSet]},  cpu_ctx, ctx_map[Constant::kTestSetFile], "dataset.test_set");
    // _dataset->valid_set = Tensor::CopyBlob(full_eid->Ptr<IdType>() + meta[Constant::kMetaNumTrainSet] + meta[Constant::kMetaNumTestSet],
    //     DataType::kI32, {meta[Constant::kMetaNumValidSet]}, cpu_ctx, ctx_map[Constant::kValidSetFile], "dataset.valid_set");
    LOG(ERROR) << "Train set size " << ToReadableSize(_dataset->train_set->NumBytes());
  }

  if (RunConfig::option_train_set_slice_mode != "") {
    const size_t full_set_size = RunConfig::unsupervised_sample ?
        meta[Constant::kMetaNumEdge] : meta[Constant::kMetaNumNode];
    // first, create an writable copy of train set
    const uint32_t origin_num_train_set = meta[Constant::kMetaNumTrainSet];
    if (RunConfig::option_train_set_slice_mode == "percent" &&
        full_set_size * RunConfig::option_train_set_percent / 100 > origin_num_train_set) {
      // expected train set exceeds original train set. so we should rebuild one from entire nodes.
      auto full_set = Tensor::Empty(kI32, {full_set_size}, CPU_CLIB(), "full_set");
      cpu::ArrangeArray(full_set->Ptr<IdType>(), full_set_size);
      _dataset->train_set = full_set;
    } else if (_dataset->train_set->Ctx().device_type != kCPU) {
      // the size of original train set meets our requirement.
      // but it is mapped and we cannot alter it, or it is in gpu.
      _dataset->train_set = Tensor::CopyTo(_dataset->train_set, CPU_CLIB());
    }

    // do a shuffle. because 1. original train set is sorted by id 2. cache file is also ranked
    // make sure train set is randomly spreaded.
    if (slice_train_should_shuffle) {
      shuffle(_dataset->train_set->Ptr<IdType>(), _dataset->train_set->Shape()[0]);
    }
    uint32_t begin = 0,end = 0;
    if (RunConfig::option_train_set_slice_mode == "percent") {
      end = full_set_size * RunConfig::option_train_set_percent / 100;
    } else if (RunConfig::option_train_set_slice_mode == "part") {
      const uint32_t part_idx = RunConfig::option_train_set_part_idx;
      const uint32_t part_num = RunConfig::option_train_set_part_num;
      const uint32_t train_set_part_size = (origin_num_train_set + part_num - 1) / part_num;
      begin = train_set_part_size * part_idx;
      end = train_set_part_size * (part_idx + 1);
      if (end > origin_num_train_set) end = origin_num_train_set;
    } else {
      CHECK(false) << "Unknown train set slice mode " << RunConfig::option_train_set_slice_mode;
    }
    meta[Constant::kMetaNumTrainSet] = end - begin;
    _dataset->train_set = Tensor::CopyBlob(
        _dataset->train_set->Ptr<uint32_t>() + begin,
        DataType::kI32, {end - begin}, CPU_CLIB(), ctx_map[Constant::kTrainSetFile], "dataset.train_set");
    std::cout << "reducing trainset from " << origin_num_train_set
              << " to " << meta[Constant::kMetaNumTrainSet]
              << " (" << meta[Constant::kMetaNumTrainSet] * 100.0 / full_set_size << ")\n";
  }

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
      case kCliquePartByDegree:
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
      case kCollCache:
      case kCollCacheIntuitive:
      case kCollCacheAsymmLink:
      case kPartitionCache:
      case kPartRepCache:
      case kRepCache:
      case kCliquePart:
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
