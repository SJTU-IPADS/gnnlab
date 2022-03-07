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
#include <parallel/numeric>
#include <parallel/algorithm>
#include <parallel_hashmap/phmap.h>
#include <thread>
#include <chrono>

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

PaGraphPartition::PaGraphPartition(const Dataset& dataset, IdType partition_num, IdType hop_num, Context sampler_ctx) 
: _hop_num(hop_num), _feat(dataset.feat), _loaded_partition(-1) {
  CHECK(_hop_num >= 1);
  LOG(INFO) << "make " << partition_num << " pgraph dataset partition, with " << _hop_num << " hop";
  IdType* partition_nodes[partition_num];
  IdType partition_nodes_num[partition_num];
  IdType* partition_trainsets[partition_num];
  IdType partition_trainsets_num[partition_num];
  // IdType* has_edge[partition_num] 
  for(int i = 0; i < partition_num; i++) {
    partition_nodes[i] = static_cast<IdType*>(Device::Get(CPU())->AllocWorkspace(
      CPU(), sizeof(IdType) * dataset.num_node, Constant::kAllocNoScale));
    partition_trainsets[i] = static_cast<IdType*>(Device::Get(CPU())->AllocWorkspace(
      CPU(), sizeof(IdType) * dataset.num_node, Constant::kAllocNoScale));
  }
  for(IdType p = 0; p < partition_num; p++) {
    partition_nodes_num[p] = 0;
    partition_trainsets_num[p] = 0;
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
    for(IdType i = 0; i < dataset.num_node; i++) {
      partition_nodes[p][i] = 0;
      partition_trainsets[p][i] = 0;
    }
  }

#if 0
  const double avg_tv = 1.0 * dataset.train_set->Shape()[0] / partition_num;
  auto score = [&](IdType p, IdType* neighbor, IdType neighbor_num) -> double {
    double comm_node = 0;
    // for(auto v : neighbor) {
    for(IdType i = 0; i < neighbor_num; i++) {
      IdType v = neighbor[i];
      comm_node += partition_trainsets[p][v];
    } 
    double balance = (avg_tv - partition_trainsets_num[p]) / (1 + partition_nodes_num[p]);
    return 1.0 * comm_node * balance;
  };

  int progress = 0;
  const IdType* trainset = static_cast<const IdType*>(dataset.train_set->Data());
  LOG(INFO) << "start partition ...";
  double get_neighbor_time = 0, score_time = 0, set_union_time = 0;
  omp_lock_t lock;
  omp_init_lock(&lock);
  IdType get_neighb_idx = 0;
  std::queue<std::tuple<IdType, IdType*, size_t>> ready_neighbor;

  for(IdType i = 0; i < dataset.train_set->Shape()[0]; i++) {
    GetNeighbor(dataset, trainset[i]);
  }
  CHECK(false);

#pragma omp parallel num_threads(RunConfig::omp_thread_num)
  {
    if(omp_get_thread_num() > 0) {
      while(true) {
        omp_set_lock(&lock);
        auto ready_num = ready_neighbor.size();
        omp_unset_lock(&lock);
        if(ready_num > 10000) {
          std::this_thread::sleep_for(std::chrono::milliseconds(300));
        }
        IdType my_idx;
#pragma omp atomic capture
        my_idx = get_neighb_idx++;
        if(my_idx > dataset.train_set->Shape()[0]) {
          break;
        }
        IdType v = trainset[my_idx];
        auto neighbor = GetNeighbor(dataset, v);
        omp_set_lock(&lock);
        ready_neighbor.push(neighbor);
        omp_unset_lock(&lock);  
      }
    } else {
      for(IdType i = 0; i < dataset.train_set->Shape()[0];) {
        Timer t0;
        omp_set_lock(&lock);
        bool ready = ready_neighbor.size() > 0;
        omp_unset_lock(&lock);
        if(!ready) {
          std::this_thread::sleep_for(std::chrono::seconds(1));
          get_neighbor_time += t0.Passed();
          continue;
        }
        omp_set_lock(&lock);
        auto neighbor = ready_neighbor.front();
        ready_neighbor.pop();
        omp_unset_lock(&lock);
          
        int cur_progress = 10000.0 * i / dataset.train_set->Shape()[0];
        if(cur_progress > progress) {
          progress = cur_progress;
          std::cout << "partition ... "<< progress / 100  << "." << progress % 100 << " %," 
                    << " get_neighbor_time " << get_neighbor_time  
                    << " score_time " << score_time
                    << " set_union_time " << set_union_time
                    << "\r" << std::flush;
        }
        
        Timer t1;
        auto vertex = std::get<0>(neighbor);
        auto neighbor_node = std::get<1>(neighbor);
        auto neighbor_num = std::get<2>(neighbor);
        double scores[partition_num];
        IdType max_score_idx[partition_num];
        IdType p = 0;
#pragma omp parallel for num_threads(partition_num)
        for(auto k = 0; k < partition_num; k++) {
          scores[k] = score(k, neighbor_node, neighbor_num);
          max_score_idx[k] = k;
        }
        std::sort(max_score_idx, max_score_idx + partition_num, [&](IdType x, IdType y) {
          return scores[x] > scores[y];
        });
        if(scores[max_score_idx[0]] == 0) {
          IdType i = 0;
          while(i < partition_num && scores[max_score_idx[i]] == 0)
            i++;
          IdType min_size = partition_nodes_num[max_score_idx[0]];
          p = max_score_idx[0];
          for(IdType j = 1; j < i; j++) {
            if(partition_nodes_num[max_score_idx[j]] < min_size) {
              min_size = partition_nodes_num[max_score_idx[j]];
              p = max_score_idx[j];
            }
          }
        }
        else if(std::abs(scores[max_score_idx[0]] - scores[max_score_idx[1]]) < 1e-6) {
          if(partition_nodes_num[max_score_idx[0]] < partition_nodes_num[max_score_idx[1]]) {
            p = max_score_idx[0];
          } else {
            p =  max_score_idx[1];
          }
        } else {
          p = max_score_idx[0];
        }
        score_time += t1.Passed();
        Timer t2;
// #pragma omp parallel for num_threads(8)
        for(IdType k = 0; k < neighbor_num; k++) {
          if(partition_nodes[p][neighbor_node[k]] == 0) {
            partition_nodes[p][neighbor_node[k]] = 1;
            partition_nodes_num[p]++;
          }
        }
        if(partition_trainsets[p][vertex] == 0) {
          partition_trainsets[p][vertex] = 1;
          partition_trainsets_num[p]++;
        }
        set_union_time += t2.Passed();
        Device::Get(CPU())->FreeWorkspace(CPU(), neighbor_node);
        i++;
      }
    }
  }
  omp_destroy_lock(&lock);
#else 

#endif
//   for(IdType i = 0; i < dataset.train_set->Shape()[0];) {
//     int cur_progress = 1000.0 * i / dataset.train_set->Shape()[0];
//     // LOG(INFO) << "progress " << i ;
//     if(cur_progress > progress) {
//       progress = cur_progress;
//       std::cout << "partition ... "<< progress / 10  << "." << progress % 10 << " %," 
//                 << " get_neighbor_time " << get_neighbor_time  
//                 << " score_time " << score_time
//                 << " set_union_time " << set_union_time
//                 << "\r" << std::flush;
//     }
//     int thread_num = i + RunConfig::omp_thread_num <= dataset.train_set->Shape()[0] ? 
//       RunConfig::omp_thread_num : dataset.train_set->Shape()[0] - i;
//     std::vector<IdType> neighbors[thread_num];
//     Timer t0;
// // #pragma omp parallel for num_threads(thread_num)
//     for(IdType j = 0; j < thread_num; j++) {
//       neighbors[j] = GetNeighbor(dataset, trainset[i + j]);
//     }
//     get_neighbor_time += t0.Passed();
//     if(i == 0) {
//       for(IdType j = 0; j < thread_num; j++) {
//         LOG(INFO) << "neighbor num " << neighbors[j].size();
//       }
//     }
//     for(IdType j = 0; j < thread_num; j++) {
//       Timer t1;
//       auto vertex = trainset[i + j];
//       auto &cur_neighbor = neighbors[j];
//       double s = score(0, cur_neighbor);
//       IdType p = 0;
//       for(auto k = 1; k < partition_num; k++) {
//         double tmp_s = score(k, cur_neighbor);
//         if(tmp_s > s) {
//           s = tmp_s, p = k;
//         }
//       }
//       score_time += t1.Passed();
//       Timer t2;
// #pragma omp parallel for num_threads(thread_num)
//       for(IdType k = 0; k < cur_neighbor.size(); k++) {
//         if(partition_nodes[p][cur_neighbor[k]] == 0) {
//           partition_nodes[p][cur_neighbor[k]] = 1;
//           partition_nodes_num[p]++;
//         }
//       }
//       if(partition_trainsets[p][vertex] == 0) {
//         partition_trainsets[p][vertex] = 1;
//         partition_trainsets_num[p]++;
//       }
//       set_union_time += t2.Passed();
//     }
//     i += thread_num;
//   }
  LOG(INFO) << "partition vertex done";
  LOG(INFO) << "partition_node_num:" 
            << std::accumulate(partition_nodes_num, 
                partition_nodes_num + partition_num, std::string{""}, 
                [](const std::string& init, const IdType first) -> std::string {
                  return init + " " + std::to_string(first);
              });
  LOG(INFO) << "partition_trainset_num:"
            << std::accumulate(partition_trainsets_num, 
                partition_trainsets_num + partition_num, std::string{""}, 
                [](const std::string& init, const IdType first) -> std::string {
                  return init + " " + std::to_string(first);
                });
  // CHECK(false);
  LOG(INFO) << "making partition ...";
  MakePartition(dataset, partition_num, 
    partition_nodes, partition_nodes_num, 
    partition_trainsets, partition_trainsets_num);

  LOG(INFO) << "making partition dataset ...";
  MakePartitionDataset(dataset);

  _iter = _partitions.begin();

  for(int p = 0; p < partition_num; p++) {
    Device::Get(CPU())->FreeWorkspace(CPU(), partition_nodes[p]);
    Device::Get(CPU())->FreeWorkspace(CPU(), partition_trainsets[p]);
  }
} 

Dataset* PaGraphPartition::GetNextPartition(Context ctx) {
  auto dataset_cpu2gpu = [&](Dataset& dataset) {
    dataset.indptr = Tensor::CopyTo(dataset.indptr, ctx);
    dataset.indices = Tensor::CopyTo(dataset.indices, ctx);
  };
  auto dataset_gpu2cpu = [](Dataset& dataset) {
    CHECK(false);
  };
  if(_iter == _partitions.end()) {
    return nullptr;
  } else {
    if(_iter != _partitions.begin()) {
      auto prev = _iter;
      prev--;
      dataset_gpu2cpu(*prev->get());
      _loaded_partition = -1;
    }
    auto partition = (_iter++)->get();
    dataset_cpu2gpu(*partition);
    _loaded_partition = _iter - _partitions.begin();
    return partition;
  }
}

TensorPtr PaGraphPartition::GetGlobalNodeId(const IdType* input_nodes, IdType input_num) {
  CHECK(_loaded_partition != -1);
  auto res_ts = Tensor::Empty(DataType::kI32, {input_num}, CPU(), "");
  auto res = static_cast<IdType*>(res_ts->MutableData());
  auto nodeId_map = static_cast<const IdType*>(_nodeId_map[_loaded_partition]->Data());
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for(IdType i = 0; i < input_num; i++) {
    res[i] = nodeId_map[input_nodes[i]];
  }
  return res_ts;
}

std::tuple<IdType, IdType*, size_t> PaGraphPartition::GetNeighbor(const Dataset& dataset, IdType vertex) {
  const IdType* indptr = static_cast<const IdType*>(dataset.indptr->Data());
  const IdType* indices = static_cast<const IdType*>(dataset.indices->Data());
  // std::vector<IdType> tmp_neighbor[_hop_num];
  IdType* tmp_neighbor[_hop_num];
  IdType tmp_neighbor_num[_hop_num] = {0};
  Timer t0;
  phmap::flat_hash_set<IdType> vis;
  IdType off = indptr[vertex];
  IdType len = indptr[vertex + 1] - indptr[vertex];
  vis.insert(vertex);
  tmp_neighbor[0] = static_cast<IdType*>(Device::Get(CPU())->AllocWorkspace(
    CPU(), sizeof(IdType) * len));
  tmp_neighbor_num[0] = len;
  for(IdType i = 0; i < len; i++) {
    tmp_neighbor[0][i] = indices[off + i];
    vis.insert(indices[i]);
  }
  LOG(INFO) << "hop 0 time " << t0.Passed();
  for(int l = 1; l < _hop_num; l++) {
    Timer t1;
    IdType max_neighbor_num = 0;
    for(IdType i = 0; i < tmp_neighbor_num[l-1]; i++) {
      IdType v = tmp_neighbor[l-1][i];
      max_neighbor_num += indptr[v+1] - indptr[v];
    }
    tmp_neighbor[l] = static_cast<IdType*>(Device::Get(CPU())->AllocWorkspace(
      CPU(), sizeof(IdType) * max_neighbor_num));
    for(IdType i = 0; i < tmp_neighbor_num[l-1]; i++) {
      IdType v = tmp_neighbor[l-1][i];
      IdType off = indptr[v];
      IdType len = indptr[v + 1] - indptr[v];
      for(IdType i = 0; i < len; i++) {
        if(vis.find(indices[off + i]) == vis.end()) {
          IdType pos = tmp_neighbor_num[l]++;
          tmp_neighbor[l][pos] = indices[off + i];
          vis.insert(indices[off + i]);
        }
      }
    }
    LOG(INFO) << "hop " << l 
              << " cur_edge " << max_neighbor_num
              << " tot_node " << vis.size()
              << " cur_neighbor_size " << tmp_neighbor_num[l] << " time " << t1.Passed();
  }
  for(int l = 0; l < _hop_num; l++) {
    Device::Get(CPU())->FreeWorkspace(CPU(), tmp_neighbor[l]);
  }

  IdType* ret = static_cast<IdType*>(Device::Get(CPU())->AllocWorkspace(
    CPU(), sizeof(IdType) * vis.size()));
  std::copy(vis.begin(), vis.end(), ret);
  return {vertex, ret, vis.size()};

//   IdType* neighbor[_hop_num];
//   IdType neighbor_num[_hop_num] = {indptr[vertex + 1] - indptr[vertex]};
//   neighbor[0] = static_cast<IdType*>(Device::Get(CPU())->AllocWorkspace(
//     CPU(), sizeof(IdType) * neighbor_num[0]));
//   IdType off = indptr[vertex];
//   IdType len = indptr[vertex + 1] - indptr[vertex];
//   for(IdType i = 0; i < len; i++) {
//     neighbor[0][i] = indices[off + i];
//   }
//   for(int l = 1; l < _hop_num; l++) {
//     IdType* tmp_cnt = static_cast<IdType*>(Device::Get(CPU())->AllocWorkspace(
//       CPU(), sizeof(IdType) * (neighbor_num[l-1] + 1)));
//     IdType* cnt = static_cast<IdType*>(Device::Get(CPU())->AllocWorkspace(
//       CPU(), sizeof(IdType) * (neighbor_num[l-1] + 1)));
//     tmp_cnt[0] = 0;
// #pragma omp parallel for num_threads(RunConfig::omp_thread_num)
//     for(IdType i = 0; i < neighbor_num[l-1]; i++) {
//       IdType v = neighbor[l - 1][i];
//       tmp_cnt[i + 1] = indptr[v + 1] - indptr[v];
//     }
//     __gnu_parallel::partial_sum(tmp_cnt, tmp_cnt + neighbor_num[l-1] + 1, cnt);
//     auto tmp_neighbor = static_cast<IdType*>(Device::Get(CPU())->AllocWorkspace(
//       CPU(), sizeof(IdType) * cnt[neighbor_num[l-1]]));
//     neighbor[l] = static_cast<IdType*>(Device::Get(CPU())->AllocWorkspace(
//       CPU(), sizeof(IdType) * cnt[neighbor_num[l-1]]));
// #pragma omp parallel for num_threads(RunConfig::omp_thread_num)
//     for(IdType i = 0; i < neighbor_num[l-1]; i++) {
//       IdType v = neighbor[l - 1][i];
//       IdType off = indptr[v];
//       IdType len = indptr[v + 1] - indptr[v];
//       IdType pos = cnt[i];
//       for(IdType j = 0; j < len; j++) {
//         tmp_neighbor[pos + j] = indices[off + j];
//       }
//     }
//     __gnu_parallel::sort(tmp_neighbor, tmp_neighbor + cnt[neighbor_num[l-1]]);
//     auto end = __gnu_parallel::unique_copy(
//       tmp_neighbor, tmp_neighbor + cnt[neighbor_num[l-1]], neighbor[l]);
//     neighbor_num[l] = end - neighbor[l];
//     Device::Get(CPU())->FreeWorkspace(CPU(), tmp_cnt);
//     Device::Get(CPU())->FreeWorkspace(CPU(), cnt);
//     Device::Get(CPU())->FreeWorkspace(CPU(), tmp_neighbor);
//   }
//   phmap::flat_hash_set<IdType> set;
//   for(IdType l = 0; l < _hop_num; l++) {
//     for(IdType i = 0; i < neighbor_num[l]; i++)
//       set.insert(neighbor[l][i]);
//     Device::Get(CPU())->FreeWorkspace(CPU(), neighbor[l]);
//   }
//   return std::vector<IdType>(set.begin(), set.end());
}

void PaGraphPartition::MakePartition(
  const Dataset& dataset, IdType partition_num, 
  IdType* partition_nodes[], IdType partition_nodes_num[], 
  IdType* partition_trainset[], IdType partition_trainset_num[]
) {
  auto indptr = static_cast<const IdType*>(dataset.indptr->Data());
  auto indices = static_cast<const IdType*>(dataset.indices->Data());

  size_t partition_tot_indices = 0;
  size_t partition_tot_indptr = 0;
  for(int p = 0; p < partition_num; p++) {
    // indptr & indices
    auto partition = std::make_unique<Dataset>();
    IdType cur_indptr_size = partition_nodes_num[p] + 1;
    IdType cur_indices_size = 0;
    // compact node
    auto node_pos_ts = Tensor::Empty(DataType::kI32, {dataset.num_node}, CPU(), "");
    IdType* node_pos = static_cast<IdType*>(node_pos_ts->MutableData());
    __gnu_parallel::partial_sum(partition_nodes[p], partition_nodes[p] + dataset.num_node, node_pos);
    CHECK(node_pos[dataset.num_node - 1] == partition_nodes_num[p]);
    auto cur_node_ts = Tensor::Empty(DataType::kI32, {partition_nodes_num[p]}, CPU(), "");
    IdType* cur_node = static_cast<IdType*>(cur_node_ts->MutableData());
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
    for(IdType i = 0; i < dataset.num_node; i++) {
      if(partition_nodes[p][i]) {
        IdType pos = node_pos[i] - 1;
        cur_node[pos] = i;
      }
    }
    LOG(INFO) << "compact node done";
    auto tmp_indptr_ts = Tensor::Empty(DataType::kI32, {cur_indptr_size}, CPU(), "");
    auto cur_indptr_ts = Tensor::Empty(DataType::kI32, {cur_indptr_size}, CPU(), 
      "partition_indptr_" + std::to_string(p));
    IdType* tmp_indptr = static_cast<IdType*>(tmp_indptr_ts->MutableData());
    IdType* cur_indptr = static_cast<IdType*>(cur_indptr_ts->MutableData());
    tmp_indptr[0] = 0;
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
    for(IdType i = 0; i < partition_nodes_num[p]; i++) {
      IdType v = cur_node[i];
      tmp_indptr[i + 1] = indptr[v + 1] - indptr[v];
    }
    __gnu_parallel::partial_sum(tmp_indptr, tmp_indptr + partition_nodes_num[p] + 1, 
      cur_indptr);
    cur_indices_size = cur_indptr[partition_nodes_num[p]];
    auto cur_indices_ts = Tensor::Empty(DataType::kI32, {cur_indices_size}, CPU(), 
      "partition_indices_" + std::to_string(p));
    IdType* cur_indices = static_cast<IdType*>(cur_indices_ts->MutableData());
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
    for(IdType i = 0; i < partition_nodes_num[p]; i++) {
      IdType v = cur_node[i];
      IdType off = indptr[v];
      IdType new_off = cur_indptr[i];
      IdType len = indptr[v + 1] - indptr[v];
      IdType new_len = cur_indptr[i + 1] - cur_indptr[i];
      CHECK(new_len == len);
      for(IdType j = 0; j < len; j++) {
        cur_indices[new_off + j] = node_pos[indices[off + j]];
      }
    }
    partition_tot_indptr += cur_indptr_size;
    partition_tot_indices += cur_indices_size;
    LOG(INFO) << "get partition indptr indices";

    auto trainset_pos_ts = Tensor::Empty(DataType::kI32, {dataset.num_node}, CPU(), "");
    IdType* trainset_pos = static_cast<IdType*>(trainset_pos_ts->MutableData());
    __gnu_parallel::partial_sum(partition_trainset[p], partition_trainset[p] + dataset.num_node,
      trainset_pos);
    CHECK(trainset_pos[dataset.num_node - 1] == partition_trainset_num[p]);
    auto trainset_ts = Tensor::Empty(DataType::kI32, {partition_trainset_num[p]}, CPU(), "");
    IdType* trainset = static_cast<IdType*>(trainset_ts->MutableData());
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
    for(IdType i = 0; i < dataset.num_node; i++) {
      if(partition_trainset[p][i]) {
        IdType pos = trainset_pos[i] - 1;
        trainset[pos] = node_pos[i];
      }
    }
    LOG(INFO) << "compact trianset done";

    partition->indptr = cur_indptr_ts;
    partition->indices = cur_indices_ts;
    partition->train_set = trainset_ts;
    _partitions.emplace_back(std::move(partition));
    _nodeId_map.push_back(node_pos_ts);
  }
  LOG(INFO) << "partition_tot_indptr " << partition_tot_indptr
            << " partition_tot_indices " << partition_tot_indices
            << " ratio " << 1.0 * partition_tot_indptr / dataset.indptr->Shape()[0]
            << " " << 1.0 * partition_tot_indices / dataset.indices->Shape()[0];
}

void PaGraphPartition::MakePartitionDataset(const Dataset& dataset) {
  auto nodeLabel = [](Dataset &partition, TensorPtr nodeId_map_ts, TensorPtr label_ts) -> TensorPtr {
    if(label_ts == nullptr || label_ts->Data() == nullptr) {
      return nullptr;
    }
    CHECK(label_ts->Shape().size() == 1);
    TensorPtr res_ts = Tensor::Empty(label_ts->Type(), {partition.num_node}, CPU(), "");
    auto nodeId_map = static_cast<const IdType*>(nodeId_map_ts->Data());
    if(label_ts->Type() == DataType::kI32) {
      auto label = static_cast<const IdType*>(label_ts->Data());
      auto res = static_cast<IdType*>(res_ts->MutableData());
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
      for(IdType i = 0; i < partition.num_node; i++) {
        IdType v = nodeId_map[i];
        res[i] = label[v];
      }
      return res_ts;
    } else if(label_ts->Type() == DataType::kI64) {
      auto label = static_cast<const uint64_t*>(label_ts->Data());
      auto res = static_cast<uint64_t*>(res_ts->MutableData());
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
      for(IdType i = 0; i < partition.num_node; i++) {
        IdType v = nodeId_map[i];
        res[i] = label[v];
      }
      return res_ts;
    } else {
      CHECK(false);
    }
  };
  for(size_t i = 0; i < _partitions.size(); i++) {
    auto&p = _partitions[i];
    p->num_edge = p->indices->Shape()[0];
    p->num_node = p->indptr->Shape()[0] - 1;
    p->num_class = dataset.num_class;
    LOG(INFO) << "partition_" << i 
              << " num_edge " << p->num_edge
              << " num_node " << p->num_node;


    CHECK(dataset.prob_table->Data() == nullptr);
    CHECK(dataset.alias_table->Data() == nullptr);
    CHECK(dataset.prob_prefix_table->Data() == nullptr);

    p->prob_table = Tensor::Null();
    p->alias_table = Tensor::Null();
    p->prob_prefix_table = Tensor::Null();

    p->label = nodeLabel(*p, _nodeId_map[i], dataset.label);
    p->in_degrees = nodeLabel(*p, _nodeId_map[i], dataset.in_degrees);
    p->out_degrees = nodeLabel(*p, _nodeId_map[i], dataset.out_degrees);
  }

  if(dataset.ranking_nodes != nullptr && dataset.ranking_nodes->Data() == nullptr) {
    CHECK(dataset.ranking_nodes->Shape()[0] == dataset.num_node);
    auto order_ts = Tensor::Empty(DataType::kI32, {dataset.num_node}, CPU(), "");
    auto order = static_cast<IdType*>(order_ts->MutableData());
    auto rank = static_cast<const IdType*>(dataset.ranking_nodes->Data());
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
    for(IdType i = 0; i < dataset.num_node; i++) {
      IdType v = rank[i];
      order[v] = i;
    }
    for(auto&p : _partitions) {
      auto rank_node_ts = Tensor::Empty(DataType::kI32, {p->num_node}, CPU(), "");
      auto rank_node = static_cast<IdType*>(rank_node_ts->MutableData());
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
      for(IdType i = 0; i < p->num_node; i++) {
        rank_node[i] = i;
      }
      __gnu_parallel::sort(rank_node, rank_node + p->num_node, [&](IdType x, IdType y) {
        return order[x] < order[y];
      });
      p->ranking_nodes = rank_node_ts;
    }
  }
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
    LOG(DEBUG) << "partition cur_node_num " << cur_node_num;
    auto partition = std::make_unique<Dataset>();
    auto nodeId_rmap = Tensor::Empty(DataType::kI32, {cur_node_num}, CPU(), "");
    auto nodeId_rmap_ptr = static_cast<IdType*>(nodeId_rmap->MutableData());
    size_t cur_indptr_size = 1;
    size_t cur_indices_size = 0;
#pragma omp parallel for num_threads(RunConfig::omp_thread_num) reduction(+:cur_indptr_size) reduction(+:cur_indices_size)
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
    LOG(DEBUG) << "subgraph size " << cur_indptr_size << " " << cur_indices_size;
    partition->indptr = Tensor::EmptyNoScale(
      DataType::kI32, {cur_indptr_size}, CPU(), "indptr");
    partition->indices = Tensor::EmptyNoScale(
      DataType::kI32, {cur_indices_size}, CPU(), "indices");
    auto cur_indptr = static_cast<IdType*>(partition->indptr->MutableData());
    auto cur_indices = static_cast<IdType*>(partition->indices->MutableData());
    // IdType tmp_indptr[cur_indptr_size];
    TensorPtr tmp_indptr_ts = Tensor::Empty(
      DataType::kI32, {cur_indptr_size}, CPU(), "");
    auto tmp_indptr = static_cast<IdType*>(tmp_indptr_ts->MutableData());
    tmp_indptr[0] = 0;
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
    for(IdType j = 0; j < cur_node_num; j++) {
      IdType k = i + j;
      size_t edge_len = indptr[k+1] - indptr[k];
      tmp_indptr[j + 1] = edge_len;
    }
    __gnu_parallel::partial_sum(tmp_indptr, tmp_indptr + cur_indptr_size, 
      cur_indptr);
    CHECK(cur_indptr[cur_node_num] == cur_indices_size);
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
    for(IdType j = 0; j < cur_node_num; j++) {
      IdType k = i + j;
      size_t edge_len = indptr[k+1] - indptr[k];
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

  Check(dataset, indptr, indices);

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

void DisjointPartition::Check(const Dataset &dataset, const IdType* h_indptr, const IdType* h_indices) {
  // auto tmp = static_cast<const IdType*>(dataset.indptr->Data());
  // CHECK(tmp[dataset.indptr->Shape()[0]-1] == dataset.num_edge);
  size_t total_nodes = 0, total_edges = 0;
  for(int p = 0; p < _partitions.size(); p++) {
    auto& dataset = _partitions[p];
    const IdType* indptr = static_cast<const IdType*>(dataset->indptr->Data());
    const IdType* indices = static_cast<const IdType*>(dataset->indices->Data());
    const size_t num_nodes = dataset->indptr->Shape()[0] - 1;
    const size_t num_edges = dataset->indices->Shape()[0];
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
    for(int i = 0; i < num_nodes; i++) {
      if(i == 0) {
        CHECK(indptr[i] == 0);
        CHECK(indptr[num_nodes] == num_edges);
      }
      CHECK(indptr[i + 1] >= indptr[i]);
    }
    TensorPtr nodeId_rmap = Tensor::Empty(DataType::kI32, 
      {_nodeId_rmap[p]->Shape()[0]}, CPU(), "");
    Device::Get(_nodeId_rmap[p]->Ctx())->CopyDataFromTo(
      _nodeId_rmap[p]->Data(), 0, nodeId_rmap->MutableData(), 0, 
      _nodeId_rmap[p]->NumBytes(), _nodeId_rmap[p]->Ctx(), CPU());
    auto nodeId_rmap_ptr = static_cast<const IdType*>(nodeId_rmap->Data());
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
    for(IdType i = 0; i < num_nodes; i++) {
      IdType new_off = indptr[i];
      IdType v = nodeId_rmap_ptr[i];
      IdType old_off = h_indptr[v];
      IdType len = indptr[i + 1] - indptr[i];
      CHECK(len == h_indptr[v + 1] - h_indptr[v]);
      for(IdType j = 0; j < len; j++) {
        CHECK(indices[new_off + j] == h_indices[old_off + j]);
      }
    }
    total_nodes += num_nodes;
    total_edges += num_edges;
  }
  CHECK(dataset.num_edge == total_edges);
  CHECK(dataset.num_node == total_nodes);
  CHECK(dataset.indptr->Shape()[0] - 1 == total_nodes);
  CHECK(dataset.indices->Shape()[0] == total_edges);
}

}
}