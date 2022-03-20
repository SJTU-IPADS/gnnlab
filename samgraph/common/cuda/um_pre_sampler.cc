#include "um_pre_sampler.h"
#include "cuda_loops.h"
#include "cuda_shuffler.h"
#include "cuda_engine.h"
#include "../run_config.h"
#include "../logging.h"
#include "../device.h"

#include <parallel/algorithm>

namespace samgraph {
namespace common {
namespace cuda {

UMPreSampler::UMPreSampler(size_t num_nodes, size_t num_step) : 
    _num_nodes(num_nodes), _num_step(num_step) {
  _freq_table_ts = Tensor::EmptyNoScale(DataType::kI32, {num_nodes}, CPU(), "freq_table");
}

void UMPreSampler::DoPreSample() {
  auto freq_table = static_cast<IdType*>(_freq_table_ts->MutableData());
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for(IdType i = 0; i < _num_nodes; i++) {
    freq_table[i] = 0;
  }
  for(int e = 0; e < RunConfig::presample_epoch; e++) {
    for(size_t i = 0; i < _num_step; i++) {
      TaskPtr task = DoShuffle();
      DoGPUSample(task);
      size_t num_nodes = task->input_nodes->Shape()[0];
      // auto input_nodes_ts = Tensor::CopyTo(task->input_nodes, CPU());
      // auto input_nodes = static_cast<const IdType*>(input_nodes_ts->Data());
      auto input_nodes = static_cast<IdType*>(Device::Get(CPU())->AllocWorkspace(
        CPU(), GetDataTypeBytes(DataType::kI32) * num_nodes, Constant::kAllocNoScale));
      Device::Get(task->input_nodes->Ctx())->CopyDataFromTo(
        task->input_nodes->Data(), 0, input_nodes, 0, 
        GetDataTypeBytes(DataType::kI32) * num_nodes, 
        task->input_nodes->Ctx(), CPU());
      auto freq_tabel = static_cast<IdType*>(_freq_table_ts->MutableData());
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
      for(size_t j = 0; j < num_nodes; j++) {
        freq_tabel[input_nodes[j]]++; 
      }
      Device::Get(CPU())->FreeWorkspace(CPU(), input_nodes);
    }
  }
  GPUEngine::Get()->GetShuffler()->Reset();
}

TensorPtr UMPreSampler::GetRankNode() const {
  auto rank_ts = Tensor::EmptyNoScale(DataType::kI32, {_num_nodes}, CPU(), "");
  auto rank = static_cast<IdType*>(rank_ts->MutableData());
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for(IdType i = 0; i < _num_nodes; i++) {
    rank[i] = i;
  }
  auto freq_table = static_cast<const IdType*>(_freq_table_ts->Data());
  __gnu_parallel::sort(rank, rank + _num_nodes, [&](IdType x, IdType y) {
    return freq_table[x] > freq_table[y];
  });
  return rank_ts;
}

} // namespace cuda
} // namespace common
} // namespace samgraph
