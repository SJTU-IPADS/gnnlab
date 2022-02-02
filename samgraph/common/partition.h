#ifndef SAMGRAPH_PARTITION_H
#define SAMGRAPH_PARTITION_H

#include <vector>
#include <unordered_set>
#include <memory>

#include "common.h"
#include "constant.h"

namespace samgraph {
namespace common {

class Partition {
 public:
  Partition(std::string data_path, IdType partition_num, IdType hop_num);
  bool Empty() { return _iter == _partitions.end(); }
  Dataset* GetNext() { return (_iter++)->get(); }

 protected:
  IdType _hop_num;
  std::vector<std::unique_ptr<Dataset>> _partitions;
  std::vector<std::unique_ptr<Dataset>>::iterator _iter;

  void MakePartition(const Dataset& dataset);
  std::unordered_set<IdType> GetNeighbor(IdType vertex, const Dataset& dataset);
  double Score(const Dataset& dataset, 
               IdType partitionId, 
               const std::vector<std::unordered_set<IdType>> &partitions,
               const std::vector<std::unordered_set<IdType>> &train_sets,
               IdType vertex, std::unordered_set<IdType>& in);
};


class DisjointPartition {
 public:
  DisjointPartition(const Dataset& dataset, IdType partition_num, Context sampler_ctx);
  std::pair<IdType, IdType> GetNewNodeId(IdType nodeId) const;
  std::pair<size_t, size_t> GetMaxPartitionSize() const;
  const Dataset& Get(IdType partitionId) const;
  const Id64Type* GetNodeIdMap() const;
  const IdType* GetNodeIdRMap(IdType partitionId) const;
  size_t Size() const;
 private:
  TensorPtr _nodeId_map; // nodeId -> {partitionId, nodeId in partition}
  std::vector<TensorPtr> _nodeId_rmap; // nodeId in partition -> nodeId
  std::vector<std::unique_ptr<Dataset>> _partitions;

  void Check(const Dataset &dataset);
};

}
};


#endif //SAMGRAPH_PARTITION_H
