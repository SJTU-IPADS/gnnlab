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
#include "dist/dist_engine.h"
#include "logging.h"
#include "profiler.h"
#include "run_config.h"
#include "timer.h"
#include "device.h"

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
#ifndef PARTITION_TEST
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

  LOG(INFO) << "unified_memory: " << RunConfig::unified_memory << " | "
            << "unified_memory_in_cpu: " << RunConfig::unified_memory_in_cpu << " | "
            << "unified_memory_overscribe_factor: " << RunConfig::unified_memory_overscribe_factor << " | "
            << "unified_memory_policy: " << static_cast<int>(RunConfig::unified_memory_policy);
  // if(RunConfig::unified_memory && 
  //    !RunConfig::unified_memory_in_cpu &&
  //    RunConfig::unified_memory_overscribe_factor > 1) {
  if(RunConfig::unified_memory &&
    (RunConfig::unified_memory_in_cpu || RunConfig::unified_memory_overscribe_factor > 1)
  ) {
    Timer sort_um_tm;
    switch (RunConfig::unified_memory_policy)
    {
    case UMPolicy::kDegree: {
      // case 1: by degree
      LOG(INFO) << "sort um dataset by Degree";
      auto order = Tensor::FromMmap(
        _dataset_path + Constant::kCacheByDegreeFile,
        DataType::kI32, {meta[Constant::kMetaNumNode]}, 
        CPU(), "order");
      SortUMDatasetBy(static_cast<const IdType*>(order->Data()));
      break;
    }
    case UMPolicy::kTrainset: {
      // case 2: by train set
      LOG(INFO) << "sort um dataset by Trainset";
      char* is_trainset = static_cast<char*>(Device::Get(CPU())->AllocWorkspace(
        CPU(), sizeof(char) * meta[Constant::kMetaNumNode], Constant::kAllocNoScale));
      auto degree_order_ts = Tensor::FromMmap(
        _dataset_path + Constant::kCacheByDegreeFile,
        DataType::kI32, {meta[Constant::kMetaNumNode]},
        CPU(), "order");
      auto degree_order = static_cast<const IdType*>(degree_order_ts->Data());
      IdType* order = static_cast<IdType*>(Device::Get(CPU())->AllocWorkspace(
        CPU(), sizeof(IdType) * meta[Constant::kMetaNumNode], Constant::kAllocScale));
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
      for(IdType i = 0; i < meta[Constant::kMetaNumNode]; i++) {
        order[i] = i;
        is_trainset[i] = false;
      }
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
      for(IdType i = 0; i < meta[Constant::kMetaNumTrainSet]; i++) {
        auto trainset = static_cast<const IdType*>(_dataset->train_set->Data());
        is_trainset[trainset[i]] = true;
      }
      __gnu_parallel::sort(order, order + meta[Constant::kMetaNumNode], [&](IdType x, IdType y){
        return std::pair<IdType, IdType>{!is_trainset[x], degree_order[x]}
          < std::pair<IdType, IdType>{!is_trainset[y], degree_order[y]};
      });
      SortUMDatasetBy(order);
      Device::Get(CPU())->FreeWorkspace(CPU(), is_trainset);
      Device::Get(CPU())->FreeWorkspace(CPU(), order);
      break;
    }
    case UMPolicy::kRandom: {
      // case 3: by random
      LOG(INFO) << "sort um dataset by Random";
      auto order = static_cast<IdType*>(Device::Get(CPU())->AllocWorkspace(
        CPU(), sizeof(IdType) * meta[Constant::kMetaNumNode], Constant::kAllocNoScale));
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
      for(IdType i = 0; i < meta[Constant::kMetaNumNode]; i++) {
        order[i] = i;
      }
      std::random_device rd;
      std::mt19937 g(rd());
      std::shuffle(order, order + meta[Constant::kMetaNumNode], g);
      SortUMDatasetBy(order);
      Device::Get(CPU())->FreeWorkspace(CPU(), order);
      break;
    }
    case UMPolicy::kPreSample: {
      LOG(INFO) << "sort um dataset by PreSample";
      CHECK(false);
      break;
    }
    // ...
    default:
      break;
    }
    LOG(INFO) << "sort um dataset " << sort_um_tm.Passed() << "secs";
  } 

  double loading_time = t.Passed();
  LOG(INFO) << "SamGraph loaded dataset(" << _dataset_path << ") successfully ("
            << loading_time << " secs)";
  LOG(DEBUG) << "dataset(" << _dataset_path << ") has "
             << _dataset->num_node << " nodes, "
             << _dataset->num_edge << " edges ";
#else
  if(partition == nullptr) {
    partition = std::make_unique<Partition>(_dataset_path, 4, 1);
  }
  _dataset = partition->GetNext();
  auto ctx_map = GetGraphFileCtx();
  if(!(ctx_map[Constant::kIndptrFile] == CPU())) {
    _dataset->indptr = Tensor::CopyTo(_dataset->indptr, ctx_map[Constant::kIndptrFile]);
  }
  if(!(ctx_map[Constant::kIndicesFile] == CPU())) {
    _dataset->indices = Tensor::CopyTo(_dataset->indices, ctx_map[Constant::kIndicesFile]);
  }
  if(RunConfig::option_empty_feat) {
    _dataset->feat = Tensor::EmptyNoScale(
          DataType::kF32, {_dataset->num_node, 1ULL << RunConfig::option_empty_feat},
          CPU(), "dataset.feat");
  }
  else {
    // TODO: copy feat to partition 
    LOG(FATAL) << "not implement yet";
  }
  _dataset->prob_table = Tensor::Null();
  _dataset->prob_prefix_table = Tensor::Null();
  _dataset->alias_table = Tensor::Null();
  _dataset->in_degrees = Tensor::Null();
  _dataset->out_degrees = Tensor::Null();
  _dataset->ranking_nodes = Tensor::Null();
  _dataset->test_set = Tensor::Null();
  _dataset->valid_set = Tensor::Null();

#endif
}

void Engine::SortUMDatasetBy(const IdType* order) {
  size_t num_nodes = _dataset->indptr->Shape()[0] - 1;
  size_t indptr_nbytes = 
    _dataset->indptr->Shape()[0] * GetDataTypeBytes(_dataset->indptr->Type());
  size_t indices_nbytes = 
    _dataset->indices->Shape()[0] * GetDataTypeBytes(_dataset->indices->Type());
  IdType* nodeIdold2new = static_cast<IdType*>(Device::Get(CPU())->AllocWorkspace(
    CPU(), num_nodes * GetDataTypeBytes(DataType::kI32), Constant::kAllocNoScale));
  IdType* tmp_indptr = static_cast<IdType*>(Device::Get(CPU())->AllocWorkspace(
    CPU(), indptr_nbytes, Constant::kAllocNoScale));
  IdType* tmp_indices = static_cast<IdType*>(Device::Get(CPU())->AllocWorkspace(
    CPU(), indices_nbytes, Constant::kAllocNoScale));
  IdType* new_indptr = static_cast<IdType*>(Device::Get(CPU())->AllocWorkspace(
    CPU(), indptr_nbytes, Constant::kAllocNoScale));
  IdType* new_indices = static_cast<IdType*>(Device::Get(CPU())->AllocWorkspace(
    CPU(), indices_nbytes, Constant::kAllocNoScale));
  
  Device::Get(_dataset->indptr->Ctx().device_type == DeviceType::kMMAP ? 
    CPU() : _dataset->indptr->Ctx())->CopyDataFromTo(
    _dataset->indptr->Data(), 0, tmp_indptr, 0, indptr_nbytes, 
    _dataset->indptr->Ctx(), CPU());
  Device::Get(_dataset->indices->Ctx().device_type == DeviceType::kMMAP ? 
    CPU() : _dataset->indices->Ctx())->CopyDataFromTo(
    _dataset->indices->Data(), 0, tmp_indices, 0, indices_nbytes,
    _dataset->indices->Ctx(), CPU());

#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for(IdType i = 0; i < num_nodes; i++) {
    nodeIdold2new[order[i]] = i;
  }

  new_indptr[0] = 0;
#pragma omp parallel for num_threads(RunConfig::omp_thread_num) 
  for(IdType i = 1; i <= num_nodes; i++) {
    IdType v = order[i-1];
    CHECK(v >= 0 && v < num_nodes);
    new_indptr[i] = tmp_indptr[v+1] - tmp_indptr[v];
  }
  __gnu_parallel::partial_sum(
    new_indptr, new_indptr + _dataset->indptr->Shape()[0], new_indptr, std::plus<IdType>());
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for(IdType i = 0; i < num_nodes; i++) {
    IdType v = order[i];
    IdType old_off = tmp_indptr[v];
    IdType new_off = new_indptr[i];
    size_t edge_len = new_indptr[i+1] - new_indptr[i];
    CHECK(edge_len == tmp_indptr[v+1] - tmp_indptr[v]);
    for(IdType j = 0; j < edge_len; j++) {
      IdType u = tmp_indices[old_off + j];
      new_indices[new_off + j] = nodeIdold2new[u];
    }
  }

// #pragma omp parallel for num_threads(RunConfig::omp_thread_num)
//   for(IdType i = 0; i < num_nodes; i++) {
//     IdType off = new_indptr[i+1] - new_indptr[i];
//     IdType old_off = tmp_indptr[order[i]+]
//   }
  auto sort_edge_values = [&](TensorPtr &values) -> void {
    if(values == nullptr || values->Data() == nullptr)
      return;
    CHECK(false);
  };
  sort_edge_values(_dataset->prob_table);
  sort_edge_values(_dataset->alias_table);
  sort_edge_values(_dataset->prob_prefix_table);

  auto sort_node_values = [&](TensorPtr &values) -> void {
    if(values == nullptr || values->Data() == nullptr)
      return;
    auto tmp_values_ts = Tensor::CopyTo(values, CPU());
    auto new_values_ts = Tensor::EmptyNoScale(
      values->Type(), values->Shape(), CPU(), values->Name());
    CHECK(tmp_values_ts->NumBytes() % tmp_values_ts->Shape()[0] == 0);
    auto per_node_nbytes = tmp_values_ts->NumBytes() / tmp_values_ts->Shape()[0];
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
    for(IdType i = 0; i < tmp_values_ts->Shape()[0]; i++) {
      size_t src = i;
      size_t dst = nodeIdold2new[i];
      memcpy(
        static_cast<char*>(new_values_ts->MutableData()) + dst * per_node_nbytes, 
        static_cast<char*>(tmp_values_ts->MutableData()) + src * per_node_nbytes, 
        per_node_nbytes  
      );
    }
    if(values->Ctx().device_type == DeviceType::kMMAP) {
      values = new_values_ts;
    } else {
      Device::Get(values->Ctx())->CopyDataFromTo(
        new_values_ts->Data(), 0, values->MutableData(), 0, values->NumBytes(),
        CPU(), values->Ctx());
    }
  };
  sort_node_values(_dataset->in_degrees);
  sort_node_values(_dataset->out_degrees);
  if(!RunConfig::option_empty_feat)
    sort_node_values(_dataset->feat);
  sort_node_values(_dataset->label);

  auto sort_nodes = [&](TensorPtr &nodes) -> void {
    if(nodes == nullptr || nodes->Data() == nullptr)
      return;
    auto tmp_nodes_ts = Tensor::CopyTo(nodes, CPU());
    auto tmp_nodes = static_cast<IdType*>(tmp_nodes_ts->MutableData());
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
    for(IdType i = 0; i < nodes->Shape()[0]; i++) {
      CHECK(tmp_nodes[i] >= 0 && tmp_nodes[i] < num_nodes);
      IdType v = tmp_nodes[i];
      IdType u = nodeIdold2new[v];
      CHECK(tmp_indptr[v+1] - tmp_indptr[v] == new_indptr[u+1] - new_indptr[u]);
      tmp_nodes[i] = nodeIdold2new[tmp_nodes[i]];
    }
    if(nodes->Ctx().device_type == DeviceType::kMMAP) {
      nodes = tmp_nodes_ts;
    } else {
      Device::Get(nodes->Ctx())->CopyDataFromTo(
        tmp_nodes, 0, nodes->MutableData(), 0, nodes->NumBytes(),
        CPU(), nodes->Ctx());
    }
  };
  sort_nodes(_dataset->ranking_nodes);
  sort_nodes(_dataset->train_set);
  sort_nodes(_dataset->valid_set);
  sort_nodes(_dataset->test_set);

  if(_dataset->indptr->Ctx().device_type == DeviceType::kMMAP) {
    _dataset->indptr = Tensor::EmptyNoScale(
      _dataset->indptr->Type(), _dataset->indptr->Shape(), CPU(), "dataset.indptr");
  }
  if(_dataset->indices->Ctx().device_type == DeviceType::kMMAP) {
    _dataset->indices = Tensor::EmptyNoScale(
      _dataset->indices->Type(), _dataset->indices->Shape(), CPU(), "dataset.indices");
  }
  Device::Get(_dataset->indptr->Ctx())->CopyDataFromTo(
    new_indptr, 0, _dataset->indptr->MutableData(), 0, indptr_nbytes,
    CPU(), _dataset->indptr->Ctx());
  Device::Get(_dataset->indices->Ctx())->CopyDataFromTo(
    new_indices, 0, _dataset->indices->MutableData(), 0, indices_nbytes,
    CPU(), _dataset->indices->Ctx());

  // free tensor
  for(auto data : {nodeIdold2new, tmp_indptr, tmp_indices, new_indptr, new_indices}) {
    Device::Get(CPU())->FreeWorkspace(CPU(), data);
  }
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
