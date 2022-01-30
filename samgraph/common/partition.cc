#include <iterator>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <iostream>
#include <sstream>
#include <random>
#include <cstring>
#include <algorithm>
#include <cassert>

#include "partition.h"
#include "logging.h"
#include "timer.h"
#include "device.h"
#include "run_config.h"

namespace samgraph {
namespace common {

Partition::Partition(std::string data_path, IdType partition_num, IdType hop_num)
: _hop_num(hop_num) {
  for(IdType i = 0; i < partition_num; i++) {
    _partitions.push_back(std::make_unique<Dataset>());
  }
  if(data_path.back() != '/')
      data_path += '/';

  LOG(INFO) << "Partition start : " << data_path << " #parti " << partition_num << " #hop " << hop_num;
  Timer parti_start; 

  std::unordered_map<std::string, size_t> meta;
  std::ifstream meta_file(data_path + Constant::kMetaFile);
  std::string line;
  while (std::getline(meta_file, line)) {
    std::istringstream iss(line);
    std::vector <std::string> kv{std::istream_iterator<std::string>{iss},
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

  Dataset dataset;
  dataset.num_edge = meta[Constant::kMetaNumEdge];
  dataset.num_node = meta[Constant::kMetaNumNode];
  dataset.num_class = meta[Constant::kMetaNumClass];

  dataset.indptr = Tensor::FromMmap(
          data_path + Constant::kIndptrFile,
          DataType::kI32,
          {meta[Constant::kMetaNumNode] + 1},
          CPU(), "dataset.indptr");
  dataset.indices = Tensor::FromMmap(
          data_path + Constant::kIndicesFile,
          DataType::kI32,
          {meta[Constant::kMetaNumEdge]},
          CPU(), "dataset.indices");

  // if(FileExist(data_path + Constant::kFeatFile)) {
  //   dataset.feat = Tensor::FromMmap(
  //           data_path + Constant::kFeatFile,
  //           DataType::kF32,
  //           {meta[Constant::kMetaNumNode], meta[Constant::kMetaFeatDim]},
  //           CPU(), "dataset.feat");
  // } else {
  //   dataset.feat = Tensor::EmptyNoScale(
  //           DataType::kF32,
  //           {meta[Constant::kMetaNumNode], meta[Constant::kMetaFeatDim]},
  //           CPU(), "dataset.feat");
  // }

  if(FileExist(data_path + Constant::kLabelFile)) {
    dataset.label = Tensor::FromMmap(
            data_path + Constant::kLabelFile,
            DataType::kI64,
            {meta[Constant::kMetaNumNode]},
            CPU(), "dataset.label");
  } else {
    dataset.label = Tensor::EmptyNoScale(
            DataType::kI64,
            {meta[Constant::kMetaNumNode]},
            CPU(), "dataset.label");
  }

  dataset.train_set = Tensor::FromMmap(
          data_path + Constant::kTrainSetFile,
          DataType::kI32,
          {meta[Constant::kMetaNumTrainSet]},
          CPU(), "dataset.train_set");
  dataset.test_set = Tensor::FromMmap(
          data_path + Constant::kTestSetFile,
          DataType::kI32,
          {meta[Constant::kMetaNumTestSet]},
          CPU(), "dataset.test_set");
  dataset.valid_set = Tensor::FromMmap(
           data_path + Constant::kValidSetFile,
           DataType::kI32,
           {meta[Constant::kMetaNumValidSet]},
           CPU(), "dataset.valid_set");

  LOG(INFO) << "partition read dataset " << parti_start.Passed();
  
  MakePartition(dataset);
  _iter = _partitions.begin();

  LOG(INFO) << "partition finish " << parti_start.Passed();
}

void Partition::MakePartition(const Dataset &dataset) {
  IdType partition_num = _partitions.size();
  std::vector<std::unordered_set<IdType>> partitions(partition_num);
  std::vector<std::unordered_set<IdType>> parti_train_sets(partition_num);

  Timer t1;
  LOG(INFO) << "train_num " << dataset.train_set->Shape()[0] 
            << " partition_num "  << partition_num;

  IdType train_num = dataset.train_set->Shape()[0];
  auto train_set = static_cast<const IdType*>(dataset.train_set->Data());

  double get_neighbor_time = 0;
  double score_time = 0;
// #pragma omp parallel for 
  for(IdType i = 0; i < train_num; i++) {
    Timer t2;
    auto in = GetNeighbor(train_set[i], dataset);
    get_neighbor_time += t2.Passed();
    t2.Reset();
    std::vector<double> scores(partition_num);
    for(IdType j = 0; j < partition_num; j++) {
      scores[j] = Score(dataset, j, partitions, parti_train_sets, train_set[i], in);
    }
    t2.Passed();
    score_time += t2.Passed();

    IdType parti = std::max_element(scores.begin(), scores.end()) - scores.begin();
    if(i < 100)
      LOG(INFO) << "node " << i << " to " << parti;
    partitions[parti].insert(in.begin(), in.end());
    parti_train_sets[parti].insert(train_set[i]);
    if(i % (train_num / 100) == 0) {
      LogMessage logger(__FILE__, __LINE__, LogLevel::INFO);
      logger << "Making partition "
                << "( " << i << "/" << train_num << " ) | ";
      for(IdType i = 0; i < partition_num; i++) {
        logger << partitions[i].size() << ", " << parti_train_sets[i].size() << " | ";
      }
      logger << " time " << t1.Passed()
             << " neighbor " << get_neighbor_time 
             << " score " << score_time;
    }
  }
  LOG(INFO) << "Finish Step 1 " << t1.Passed();
  for(IdType i = 0; i < partition_num; i++) {
    LOG(INFO) << "parti " << i 
              << " train_set_num " << parti_train_sets[i].size()
              << " partition_size " << partitions[i].size();
  }
  // create partition datasets
  auto idx_map_tp = Tensor::EmptyNoScale(
            DataType::kI32, {dataset.num_node}, CPU(), "idx_map");
  auto idx_map = static_cast<IdType*>(idx_map_tp->MutableData());
  auto indptr = static_cast<const IdType*> (dataset.indptr->Data());
  auto indices = static_cast<const IdType*> (dataset.indices->Data());
  auto label = static_cast<const uint64_t*>(dataset.label->Data());
  for(IdType i = 0; i < partitions.size(); i++) {
    size_t parti_indptr_size = partitions[i].size();
    size_t parti_indices_size = 0;
    IdType j = 0;
    for(IdType v : partitions[i]) {
      idx_map[v] = j++;
      parti_indices_size += indptr[v+1] - indptr[v];
    }
    LOG(INFO) << "parti " << i << " indices "<< parti_indices_size;
    auto& parti_dataset = *_partitions[i];
    parti_dataset.indptr = Tensor::EmptyNoScale(
            DataType::kI32, {parti_indptr_size + 1}, CPU(), "dataset.indptr");
    parti_dataset.indices = Tensor::EmptyNoScale(
            DataType::kI32, {parti_indices_size}, CPU(), "dataset.indices");
    parti_dataset.train_set = Tensor::EmptyNoScale(
            DataType::kI32, {parti_train_sets[i].size()}, CPU(), "dataset.train_set");
    parti_dataset.label = Tensor::EmptyNoScale(
            DataType::kI64, {parti_indptr_size}, CPU(), "dataset.label");
    auto parti_indptr = static_cast<IdType*>(parti_dataset.indptr->MutableData());
    auto parti_indices = static_cast<IdType*>(parti_dataset.indices->MutableData());
    auto parti_train_set = static_cast<IdType*>(parti_dataset.train_set->MutableData());
    auto parti_label = static_cast<uint64_t*>(parti_dataset.label->MutableData());
    // copy partition graph 
    j = 0;
    parti_indptr[0] = 0;
    for(auto v : partitions[i]) {
      assert(j == idx_map[v]);
      IdType pos = parti_indptr[j++];
      for(IdType k = indptr[v]; k < indptr[v+1]; k++) {
        if(!partitions[i].count(indices[k])) continue;
        parti_indices[pos++] = idx_map[indices[k]];
      }
      parti_indptr[j] = pos;
    }
    // memcpy(parti_train_set, parti_train_sets[i].data(), sizeof(IdType) * parti_train_sets[i].size());
    j = 0;
    for(auto v : parti_train_sets[i]) {
      parti_train_set[j++] = idx_map[v];
    }
    for(auto v : partitions[i]) {
      parti_label[idx_map[v]] = label[v];
    }
    parti_dataset.num_node = parti_dataset.indptr->Shape()[0] - 1;
    parti_dataset.num_edge = parti_dataset.indices->Shape()[0];
    parti_dataset.num_class = dataset.num_class;
  }
}

std::unordered_set<IdType> Partition::GetNeighbor(IdType vertex, const Dataset &dataset) {
  std::unordered_set<IdType> neighbor;
  auto indptr = static_cast<const IdType*>(dataset.indptr->Data());
  auto indices = static_cast<const IdType*>(dataset.indices->Data());
  std::queue<std::pair<IdType, IdType>> que;
  que.push({vertex, 0});
  while(!que.empty()) {
    auto v = que.front();
    que.pop();
    neighbor.insert(v.first);
    if(v.second >= _hop_num) continue;
    for(IdType i = indptr[v.first]; i < indptr[v.first + 1]; i++) {
      que.push({indices[i], v.second + 1});
    }
  }
  return neighbor;
}

double Partition::Score(const Dataset& dataset, 
                        IdType partitionId,
                        const std::vector<std::unordered_set<IdType>> &partitions,
                        const std::vector<std::unordered_set<IdType>> &train_sets,
                        IdType vertex, std::unordered_set<IdType> &in) {
  IdType comm_node = 0;
  for(auto v : in) {
    comm_node += train_sets[partitionId].count(v);
  }
  comm_node = std::max(comm_node, 1U);
  double train_avg = 1.0 * dataset.train_set->Shape()[0] / partitions.size();
  double balance = 1.0 * (train_avg - train_sets[partitionId].size()) / std::max(partitions[partitionId].size(), (size_t)1);
  return 1.0 * comm_node * balance;

  // std::mt19937 gen(2021);
  // std::uniform_real_distribution<> dist(0, 1);
  // return dist(gen);
}

DisjointPartition::DisjointPartition(const Dataset& dataset, IdType partition_num, Context sampler_ctx) {
  LOG(INFO) << "make " << partition_num << " dataset partition";
  auto nodeId_map = Tensor::EmptyNoScale(
    DataType::kI64, {dataset.num_node}, CPU(), "nodeId_map");
  auto nodeId_map_ptr = static_cast<Id64Type*>(nodeId_map->MutableData());

  auto indptr = static_cast<IdType*>(Device::Get(CPU())->AllocWorkspace(
    CPU(), dataset.indptr->NumBytes(), Constant::kAllocNoScale));
  auto indices = static_cast<IdType*>(Device::Get(CPU())->AllocWorkspace(
    CPU(), dataset.indices->NumBytes(), Constant::kAllocNoScale));
  Device::Get(dataset.indptr->Ctx())->CopyDataFromTo(
    dataset.indptr->Data(), 0, indptr, 0, dataset.indptr->NumBytes(), 
    dataset.indptr->Ctx(), CPU());
  Device::Get(dataset.indices->Ctx())->CopyDataFromTo(
    dataset.indices->Data(), 0, indices, 0, dataset.indices->NumBytes(),
    dataset.indices->Ctx(), CPU());

  IdType per_partition_node_num = dataset.num_node / partition_num;
  for(IdType i = 0, p = 0; i < dataset.num_node && p < partition_num; p++) {
    IdType cur_node_num = per_partition_node_num + (p < dataset.num_node % partition_num);
    auto partition = std::make_unique<Dataset>();
    auto nodeId_rmap = Tensor::Empty(DataType::kI32, {cur_node_num}, CPU(), "");
    auto nodeId_rmap_ptr = static_cast<IdType*>(nodeId_rmap->MutableData());
    size_t cur_indptr_size = 1;
    size_t cur_indices_size = 0;
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
    for(IdType j = 0; j < cur_node_num; j++) {
      IdType k = i + j;
      auto cur_nodeId_map = reinterpret_cast<IdType*>(&nodeId_map_ptr[k]);
      cur_nodeId_map[0] = p;
      cur_nodeId_map[1] = j;
      nodeId_rmap_ptr[j] = k;
      size_t edge_len = indptr[k+1] - indptr[k];
      cur_indptr_size += 1;
      cur_indices_size += edge_len;
    }
    partition->indptr = Tensor::EmptyNoScale(
      DataType::kI32, {cur_indptr_size}, CPU(), "indptr");
    partition->indices = Tensor::EmptyNoScale(
      DataType::kI32, {cur_indices_size}, CPU(), "indices");
    auto cur_indptr = static_cast<IdType*>(partition->indptr->MutableData());
    auto cur_indices = static_cast<IdType*>(partition->indices->MutableData());
    cur_indptr[0] = 0;
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
    for(IdType j = 0; j < cur_node_num; j++) {
      IdType k = i + j;
      size_t edge_len = indptr[k+1] - indptr[k];
      cur_indptr[j+1] = cur_indptr[j] + edge_len;
      size_t old_off = indptr[k];
      size_t new_off = cur_indptr[j];
      for(IdType l = 0; l < edge_len; l++) {
        cur_indices[new_off + l] = indices[old_off + l];
      }
    }
    i += cur_node_num;
    _partitions.emplace_back(std::move(partition));
    _nodeId_rmap.push_back(Tensor::CopyTo(nodeId_rmap, sampler_ctx));
  }

  _nodeId_map = Tensor::CopyTo(nodeId_map, sampler_ctx);
  CHECK(dataset.alias_table == nullptr || dataset.alias_table->Data() == nullptr);
  CHECK(dataset.prob_table == nullptr || dataset.prob_table->Data() == nullptr);
  CHECK(dataset.prob_prefix_table == nullptr || dataset.prob_prefix_table->Data() == nullptr);

  Check();

  Device::Get(CPU())->FreeWorkspace(CPU(), indptr);
  Device::Get(CPU())->FreeWorkspace(CPU(), indices);
}

std::pair<IdType, IdType> DisjointPartition::GetNewNodeId(IdType nodeId ) const {
  auto nodeId_map_ptr = static_cast<const Id64Type*>(_nodeId_map->Data());
  auto cur_nodeId_map = reinterpret_cast<const IdType*>(&nodeId_map_ptr[nodeId]);
  return {cur_nodeId_map[0], cur_nodeId_map[1]};
}

std::pair<size_t, size_t> DisjointPartition::GetMaxPartitionSize() const {
  return std::accumulate(_partitions.begin(), _partitions.end(), std::pair<size_t, size_t>{0, 0}, 
    [](std::pair<size_t, size_t> init, const std::unique_ptr<Dataset> &ds) {
      init.first = std::max(init.first, ds->indptr->Shape()[0]);
      init.second = std::max(init.second, ds->indices->Shape()[0]);
      return init;
  });
}

const Dataset& DisjointPartition::Get(IdType partitionId) const {
  return *_partitions[partitionId].get();
}

const Id64Type* DisjointPartition::GetNodeIdMap() const {
  return static_cast<const Id64Type*>(_nodeId_map->Data());
}

const IdType* DisjointPartition::GetNodeIdRMap(IdType partitionId) const {
  return static_cast<const IdType*>(_nodeId_rmap[partitionId]->Data());
}

size_t DisjointPartition::Size() const {
  return _partitions.size();
}

void DisjointPartition::Check() {

}

}
}