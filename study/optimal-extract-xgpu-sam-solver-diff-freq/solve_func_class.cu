#include "common/common.h"
#include "common/cuda/cub_sort_wrapper.h"
#include "common/cuda/cuda_utils.h"
#include "common/run_config.h"
#include "common/cpu/mmap_cpu_device.h"
#include "common/coll_cache/optimal_solver.h"
#include "common/coll_cache/ndarray.h"
#include "solve_func_class.h"
#include <omp.h>
#include <unistd.h>
#include <iostream>

namespace samgraph{
namespace common{
namespace coll_cache{

#define FOR_LOOP(iter, len) for (uint32_t iter = 0; iter < (len); iter++)

void RepSolver::Solve(std::vector<int> device_to_stream,
                      std::vector<int> device_to_cache_percent,
                      std::string mode, double T_local, double T_remote,
                      double T_cpu) {
  CHECK(std::accumulate(device_to_stream.begin(), device_to_stream.end(), 0, std::plus<>()) == 0);
  const int num_device = 2;
  const int num_block = 2;
  const IdType num_node = stream_freq_list->Shape()[1];
  // the freq list must already be sorted
  // now calculate the boundary
  const IdType num_cached_nodes = num_node * (device_to_cache_percent[0] / (double)100);

  LOG(ERROR) << "num_cached_nodes = " << num_cached_nodes;

  const IdType * freq_array = stream_freq_list->Ptr<IdType>();

  const IdType replicate_size = num_cached_nodes;
  const IdType cpu_size = num_node - replicate_size;

  double rep_w = 0, cpu_w = 0, total_w = 0;

#pragma omp parallel for num_threads(RunConfig::omp_thread_num) reduction(+ : rep_w, cpu_w, total_w)
  for (IdType rank = 0; rank < num_node; rank++) {
    IdType node_id = stream_id_list->Ptr<IdType>()[rank];
    if (rank < replicate_size) {
      nid_to_block->Ptr<IdType>()[node_id] = 0;
      rep_w += freq_array[rank];
    } else {
      nid_to_block->Ptr<IdType>()[node_id] = 1;
      cpu_w += freq_array[rank];
    }
    total_w += freq_array[rank];
  }
  double local_w = rep_w;
  std::cout << "coll_cache:optimal_rep_storage=" << replicate_size / (double)num_node << "\n";
  std::cout << "coll_cache:optimal_part_storage=" << 0 << "\n";
  std::cout << "coll_cache:optimal_cpu_storage=" << cpu_size / (double)num_node << "\n";
  std::cout << "coll_cache:optimal_local_storage=" << replicate_size / (double)num_node << "\n";
  std::cout << "coll_cache:optimal_remote_storage=" << 0 << "\n";
  std::cout << "coll_cache:optimal_local_rate=" << local_w / total_w << "\n";
  std::cout << "coll_cache:optimal_remote_rate=" << 0 << "\n";
  std::cout << "coll_cache:optimal_cpu_rate=" << cpu_w / total_w << "\n";
  std::cout << "z=" << local_w * 100 / num_node * T_local + cpu_w * 100 / num_node * T_cpu << "\n";
}


void MultiStreamSolverBase::Build(TensorPtr stream_id_list,
                                  TensorPtr stream_freq_list,
                                  std::vector<int> device_to_stream,
                                  const IdType num_node,
                                  const TensorPtr nid_to_block_tensor) {
  this->stream_id_list = stream_id_list;
  this->stream_freq_list = stream_freq_list;
  this->nid_to_block = nid_to_block_tensor;
  CHECK_EQ(stream_id_list->Shape().size(), 2);
  CHECK_EQ(stream_freq_list->Shape().size(), 2);
  CHECK_EQ(stream_id_list->Shape(), stream_freq_list->Shape());

  TensorView<IdType> stream_id_list_view(stream_id_list);
  IdType num_stream = stream_id_list->Shape()[0];
  auto cpu_ctx = CPU(CPU_CLIB_MALLOC_DEVICE);

  /**
   * Map each node to a rank for each stream.
   * Nodes with same rank for every stream forms a block.
   */
  nid_to_rank_tensor  = Tensor::Empty(kI32, {num_node, num_stream}, cpu_ctx, "coll_cache.nid_to_rank");
  TensorView<uint32_t> nid_to_rank(nid_to_rank_tensor);
  LOG(INFO) << "mapping nid to rank...";
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for (uint32_t orig_rank = 0; orig_rank < num_node; orig_rank++) {
    for (uint32_t stream_idx = 0; stream_idx < num_stream; stream_idx++) {
      uint32_t nid = stream_id_list_view[stream_idx][orig_rank].ref();
      nid_to_rank[nid][stream_idx].ref() = orig_rank;
    }
  }
}

void SelfishSolver::Solve(std::vector<int> device_to_stream,
                          std::vector<int> device_to_cache_percent,
                          std::string mode, double T_local, double T_remote,
                          double T_cpu) {
  IdType num_node = stream_id_list->Shape()[1];
  IdType num_stream = stream_id_list->Shape()[0];
  IdType num_device = device_to_stream.size();
  auto cpu_ctx = CPU(CPU_CLIB_MALLOC_DEVICE);

  TensorView<IdType> stream_id_list_view(stream_id_list);
  TensorView<IdType> stream_freq_list_view(stream_freq_list);
  TensorView<uint32_t> nid_to_rank(nid_to_rank_tensor);

  /**
   * map node to block
   */
  const IdType num_block = 1 << num_device;
  const IdType num_cached_nodes = num_node * (device_to_cache_percent[0] / (double)100);
  LOG(ERROR) << "num_cached_nodes = " << num_cached_nodes;

#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for (IdType nid = 0; nid < num_node; nid++) {
    uint8_t nid_storage = 0;
    for (IdType dev_id = 0; dev_id < num_device; dev_id++) {
      IdType stream_id = device_to_stream[dev_id];
      IdType rank = nid_to_rank[nid][stream_id].ref();
      if (rank < num_cached_nodes) {
        nid_storage |= 1 << dev_id;
      }
    }
    nid_to_block->Ptr<IdType>()[nid] = nid_storage;
  }

  /**
   * count each block's density & freq, based on 
   * I think this may be take out to be a universal function?
   */
  auto block_density_tensor = Tensor::Empty(kF64, {num_block}, cpu_ctx, "coll_cache.block_density_tensor");
  auto block_freq_tensor    = Tensor::Empty(kF64, {num_block, num_stream}, cpu_ctx, "coll_cache.block_freq_tensor");
  auto block_density_array = TensorView<double>(block_density_tensor);
  auto block_freq_array = TensorView<double>(block_freq_tensor);
  for (IdType block_id = 0; block_id < num_block; block_id ++) {
    block_density_array[block_id].ref() = 0;
    for (IdType stream_idx = 0; stream_idx < num_stream; stream_idx++) {
      block_freq_array[block_id][stream_idx].ref() = 0;
    }
  }

#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for (int thread_id = 0; thread_id < RunConfig::omp_thread_num; thread_id++) {
    auto local_block_density_tensor = Tensor::Empty(kF64, {num_block}, cpu_ctx, "coll_cache.block_density_tensor");
    auto local_block_freq_tensor    = Tensor::Empty(kF64, {num_block, num_stream}, cpu_ctx, "coll_cache.block_freq_tensor");
    auto local_block_freq_array = TensorView<double>(local_block_freq_tensor);
    for (IdType block_id = 0; block_id < num_block; block_id ++) {
      local_block_density_tensor->Ptr<double>()[block_id] = 0;
      for (IdType stream_idx = 0; stream_idx < num_stream; stream_idx++) {
        local_block_freq_array[block_id][stream_idx].ref() = 0;
      }
    }
    for (IdType nid = omp_get_thread_num(); nid < num_node; nid += RunConfig::omp_thread_num) {
      IdType block_id = nid_to_block->CPtr<IdType>()[nid];
      local_block_density_tensor->Ptr<double>()[block_id] += 1;
      for (IdType stream_idx = 0; stream_idx < num_stream; stream_idx++) {
        uint32_t orig_rank = nid_to_rank[nid][stream_idx].ref();
        double freq = stream_freq_list_view[stream_idx][orig_rank].ref();
        // assign all zero freq a minimal freq to handle touched node << cache space
        freq = std::max(freq, 1e-3);
        local_block_freq_array[block_id][stream_idx].ref() += freq;
      }
    }
    #pragma omp critical
    {
      for (IdType block_id = 0; block_id < num_block; block_id ++) {
        block_density_array[block_id].ref() += local_block_density_tensor->CPtr<double>()[block_id];
        for (IdType stream_idx = 0; stream_idx < num_stream; stream_idx++) {
          block_freq_array[block_id][stream_idx].ref() += local_block_freq_array[block_id][stream_idx].ref();
        }
      }
    }
  }

  std::vector<double> local_w(num_device, 0);
  std::vector<double> remote_w(num_device, 0);
  std::vector<double> cpu_w(num_device, 0);
  std::vector<double> total_w(num_device, 0);
  std::vector<double> z(num_device, 0);

  std::cout << "coll_cache:optimal_cpu_storage=" << block_density_array[0].ref() / (double)num_node << "\n";
  std::cout << "coll_cache:optimal_local_storage=" << num_cached_nodes / (double)num_node << "\n";
  std::cout << "coll_cache:optimal_remote_storage=" << 1 - block_density_array[0].ref()/(double)num_node - num_cached_nodes / (double)num_node << "\n";

  for (uint32_t block_id = 0; block_id < num_block; block_id++) {
    if (block_density_array[block_id].ref() == 0) continue; 
    for (uint32_t stream_id = 0; stream_id < num_stream; stream_id++) {
      block_freq_array[{block_id,stream_id}].ref() /= block_density_array[block_id].ref() ;
    }
    block_density_array[block_id].ref() *= 100/(double)num_node ;
    for (IdType dev_id = 0; dev_id < num_device; dev_id++) {
      IdType stream_id = device_to_stream[dev_id];
      if (block_id & (1 << dev_id)) {
        local_w[dev_id] += block_freq_array[block_id][stream_id].ref() * block_density_array[block_id].ref();
      } else if (block_id != 0) {
        remote_w[dev_id] += block_freq_array[block_id][stream_id].ref() * block_density_array[block_id].ref();
      } else {
        cpu_w[dev_id] += block_freq_array[block_id][stream_id].ref() * block_density_array[block_id].ref();
      }
      total_w[dev_id] += block_freq_array[block_id][stream_id].ref() * block_density_array[block_id].ref();
    }
  }

  std::cout << "coll_cache:optimal_local_rate=";
  FOR_LOOP(dev_id, num_device) { std::cout << local_w[dev_id] / total_w[dev_id] << ","; }
  std::cout << "\n";
  std::cout << "coll_cache:optimal_remote_rate=";
  FOR_LOOP(dev_id, num_device) { std::cout << remote_w[dev_id] / total_w[dev_id] << ","; }
  std::cout << "\n";
  std::cout << "coll_cache:optimal_cpu_rate=";
  FOR_LOOP(dev_id, num_device) { std::cout << cpu_w[dev_id] / total_w[dev_id] << ","; }
  std::cout << "\n";
  std::cout << "z=";
  FOR_LOOP(dev_id, num_device) { std::cout << local_w[dev_id] * T_local + remote_w[dev_id] * T_remote + cpu_w[dev_id] * T_cpu << ","; }
  std::cout << "\n";
}

void PartRepMultiStream::Build(TensorPtr stream_id_list, TensorPtr stream_freq_list, std::vector<int> device_to_stream, const IdType num_node, const TensorPtr nid_to_block_tensor) {
  MultiStreamSolverBase::Build(stream_id_list, stream_freq_list, device_to_stream, num_node, nid_to_block_tensor);
  /**
   * aggregate a total freq of all device
   */
  IdType num_device = device_to_stream.size();
  auto cpu_ctx = CPU(CPU_CLIB_MALLOC_DEVICE);
  auto gpu_device = Device::Get(GPU(0));
  TensorView<IdType> stream_freq_list_view(stream_freq_list);
  TensorView<uint32_t> nid_to_rank(nid_to_rank_tensor);
  global_rank_to_freq = Tensor::Empty(kF64, {num_node}, cpu_ctx, "");
  global_rank_to_nid = Tensor::Empty(kI32, {num_node}, cpu_ctx, "");
  {
    double* nid_to_global_freq = gpu_device->AllocArray<double>(GPU(0), num_node);
    double* nid_to_global_freq_out = gpu_device->AllocArray<double>(GPU(0), num_node);
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
    for (IdType nid = 0; nid < num_node; nid++) {
      double sum = 0;
      for (IdType dev_id = 0; dev_id < num_device; dev_id++) {
        IdType stream_id = device_to_stream[dev_id];
        IdType rank = nid_to_rank[nid][stream_id].ref();
        sum += stream_freq_list_view[stream_id][rank].ref();
      }
      global_rank_to_freq->Ptr<double>()[nid] = sum;
    }
    gpu_device->CopyDataFromTo(global_rank_to_freq->CPtr<double>(), 0, nid_to_global_freq, 0, global_rank_to_freq->NumBytes(), cpu_ctx, GPU(0));
    gpu_device->SyncDevice(GPU(0));
    IdType* nid = gpu_device->AllocArray<IdType>(GPU(0), num_node);
    IdType* nid_out = gpu_device->AllocArray<IdType>(GPU(0), num_node);
    cuda::ArrangeArray<IdType>(nid, num_node);
    gpu_device->SyncDevice(GPU(0));
    cuda::CubSortPairDescending(nid_to_global_freq, nid_to_global_freq_out, nid, nid_out, num_node, GPU(0));
    gpu_device->SyncDevice(GPU(0));
    gpu_device->CopyDataFromTo(nid_to_global_freq_out, 0, global_rank_to_freq->Ptr<double>(), 0, global_rank_to_freq->NumBytes(), GPU(0), cpu_ctx);
    gpu_device->CopyDataFromTo(nid_out, 0, global_rank_to_nid->Ptr<IdType>(), 0, global_rank_to_nid->NumBytes(), GPU(0), cpu_ctx);
    gpu_device->SyncDevice(GPU(0));
    gpu_device->FreeWorkspace(GPU(0), nid_to_global_freq_out);
    gpu_device->FreeWorkspace(GPU(0), nid_to_global_freq);
    gpu_device->FreeWorkspace(GPU(0), nid_out);
    gpu_device->FreeWorkspace(GPU(0), nid);
  }
}

void PartRepMultiStream::Solve(std::vector<int> device_to_stream,
                          std::vector<int> device_to_cache_percent,
                          std::string mode, double T_local, double T_remote,
                          double T_cpu) {
  IdType num_node = stream_id_list->Shape()[1];
  IdType num_stream = stream_id_list->Shape()[0];
  IdType num_device = device_to_stream.size();
  auto cpu_ctx = CPU(CPU_CLIB_MALLOC_DEVICE);
  const IdType num_cached_nodes = num_node * (device_to_cache_percent[0] / (double)100);

  TensorView<IdType> stream_id_list_view(stream_id_list);
  TensorView<IdType> stream_freq_list_view(stream_freq_list);
  TensorView<uint32_t> nid_to_rank(nid_to_rank_tensor);

  const IdType partition_size = (num_device == 1) ? num_cached_nodes : std::min(num_cached_nodes, (num_node - num_cached_nodes)/(num_device-1));
  const IdType replicate_size = num_cached_nodes - partition_size;
  CHECK_LE(replicate_size + partition_size * num_device, num_node);
  const IdType cpu_size = num_node - replicate_size - partition_size * num_device;

  std::vector<double> local_w(num_device, 0);
  std::vector<double> remote_w(num_device, 0);
  std::vector<double> cpu_w(num_device, 0);
  std::vector<double> total_w(num_device, 0);
  std::vector<double> z(num_device, 0);

  for (IdType dev_id = 0; dev_id < num_device; dev_id++) {
    double rep_w_ = 0, cpu_w_ = 0, total_w_ = 0;
    IdType stream_idx = device_to_stream[dev_id];
    #pragma omp parallel for num_threads(RunConfig::omp_thread_num) reduction(+ : rep_w_, cpu_w_, total_w_)
    for (IdType rank = 0; rank < num_node; rank++) {
      IdType node_id = global_rank_to_nid->Ptr<IdType>()[rank];
      IdType original_rank = nid_to_rank[node_id][stream_idx].ref();
      IdType original_freq = stream_freq_list_view[stream_idx][original_rank].ref();
      if (rank < replicate_size) {
        rep_w_ += original_freq;
      } else if (rank < replicate_size + partition_size * num_device) {
      } else {
        cpu_w_ += original_freq;
      }
      total_w_ += original_freq;
    }
    double partition_w_ = total_w_ - cpu_w_ - rep_w_;
    local_w[dev_id] = rep_w_ + partition_w_ / num_device;
    remote_w[dev_id] = partition_w_ / num_device * (num_device - 1);
    cpu_w[dev_id] = cpu_w_;
    total_w[dev_id] = total_w_;
    z[dev_id] = local_w[dev_id] * 100 / num_node * T_local + remote_w[dev_id] * 100 / num_node * T_remote + cpu_w[dev_id] * 100 / num_node * T_cpu;
  }

  std::cout << "coll_cache:optimal_cpu_storage=" << cpu_size / (double)num_node << "\n";
  std::cout << "coll_cache:optimal_local_storage=" << num_cached_nodes / (double)num_node << "\n";
  std::cout << "coll_cache:optimal_remote_storage=" << 1 - cpu_size/(double)num_node - num_cached_nodes / (double)num_node << "\n";

  std::cout << "coll_cache:optimal_local_rate=";
  FOR_LOOP(dev_id, num_device) { std::cout << local_w[dev_id] / total_w[dev_id] << ","; }
  std::cout << "\n";
  std::cout << "coll_cache:optimal_remote_rate=";
  FOR_LOOP(dev_id, num_device) { std::cout << remote_w[dev_id] / total_w[dev_id] << ","; }
  std::cout << "\n";
  std::cout << "coll_cache:optimal_cpu_rate=";
  FOR_LOOP(dev_id, num_device) { std::cout << cpu_w[dev_id] / total_w[dev_id] << ","; }
  std::cout << "\n";
  std::cout << "z=";
  FOR_LOOP(dev_id, num_device) { std::cout << z[dev_id] << ","; }
  std::cout << "\n";
}

void PartitionMultiStream::Solve(std::vector<int> device_to_stream,
                          std::vector<int> device_to_cache_percent,
                          std::string mode, double T_local, double T_remote,
                          double T_cpu) {
  IdType num_node = stream_id_list->Shape()[1];
  IdType num_stream = stream_id_list->Shape()[0];
  IdType num_device = device_to_stream.size();
  auto cpu_ctx = CPU(CPU_CLIB_MALLOC_DEVICE);
  const IdType num_cached_nodes = num_node * (device_to_cache_percent[0] / (double)100);

  TensorView<IdType> stream_id_list_view(stream_id_list);
  TensorView<IdType> stream_freq_list_view(stream_freq_list);
  TensorView<uint32_t> nid_to_rank(nid_to_rank_tensor);

  const IdType partition_size = std::min(num_cached_nodes, num_node / num_device);
  const IdType replicate_size = 0;
  CHECK_LE(replicate_size + partition_size * num_device, num_node);
  const IdType cpu_size = num_node - replicate_size - partition_size * num_device;

  std::vector<double> local_w(num_device, 0);
  std::vector<double> remote_w(num_device, 0);
  std::vector<double> cpu_w(num_device, 0);
  std::vector<double> total_w(num_device, 0);
  std::vector<double> z(num_device, 0);

  for (IdType dev_id = 0; dev_id < num_device; dev_id++) {
    double rep_w_ = 0, cpu_w_ = 0, total_w_ = 0;
    IdType stream_idx = device_to_stream[dev_id];
    #pragma omp parallel for num_threads(RunConfig::omp_thread_num) reduction(+ : rep_w_, cpu_w_, total_w_)
    for (IdType rank = 0; rank < num_node; rank++) {
      IdType node_id = global_rank_to_nid->Ptr<IdType>()[rank];
      IdType original_rank = nid_to_rank[node_id][stream_idx].ref();
      IdType original_freq = stream_freq_list_view[stream_idx][original_rank].ref();
      if (rank < replicate_size) {
        rep_w_ += original_freq;
      } else if (rank < replicate_size + partition_size * num_device) {
      } else {
        cpu_w_ += original_freq;
      }
      total_w_ += original_freq;
    }
    double partition_w_ = total_w_ - cpu_w_ - rep_w_;
    local_w[dev_id] = rep_w_ + partition_w_ / num_device;
    remote_w[dev_id] = partition_w_ / num_device * (num_device - 1);
    cpu_w[dev_id] = cpu_w_;
    total_w[dev_id] = total_w_;
    z[dev_id] = local_w[dev_id] * 100 / num_node * T_local + remote_w[dev_id] * 100 / num_node * T_remote + cpu_w[dev_id] * 100 / num_node * T_cpu;
  }

  std::cout << "coll_cache:optimal_cpu_storage=" << cpu_size / (double)num_node << "\n";
  std::cout << "coll_cache:optimal_local_storage=" << num_cached_nodes / (double)num_node << "\n";
  std::cout << "coll_cache:optimal_remote_storage=" << 1 - cpu_size/(double)num_node - num_cached_nodes / (double)num_node << "\n";

  std::cout << "coll_cache:optimal_local_rate=";
  FOR_LOOP(dev_id, num_device) { std::cout << local_w[dev_id] / total_w[dev_id] << ","; }
  std::cout << "\n";
  std::cout << "coll_cache:optimal_remote_rate=";
  FOR_LOOP(dev_id, num_device) { std::cout << remote_w[dev_id] / total_w[dev_id] << ","; }
  std::cout << "\n";
  std::cout << "coll_cache:optimal_cpu_rate=";
  FOR_LOOP(dev_id, num_device) { std::cout << cpu_w[dev_id] / total_w[dev_id] << ","; }
  std::cout << "\n";
  std::cout << "z=";
  FOR_LOOP(dev_id, num_device) { std::cout << z[dev_id] << ","; }
  std::cout << "\n";
}

void CliqueGlobalFreqSolver::Solve(std::vector<int> device_to_stream,
                         std::vector<int> device_to_cache_percent,
                         std::string mode, double T_local, double T_remote,
                         double T_cpu) {
  int num_device = device_to_stream.size();
  CHECK(num_device % clique_size == 0);
  device_to_stream.resize(clique_size);
  device_to_cache_percent.resize(clique_size);
  PartRepMultiStream::Solve(device_to_stream, device_to_cache_percent, mode, T_local, T_remote, T_cpu);
};
void CliqueLocalFreqSolver::Build(TensorPtr stream_id_list,
                                  TensorPtr stream_freq_list,
                                  std::vector<int> device_to_stream,
                                  const IdType num_node,
                                  const TensorPtr nid_to_block_tensor) {
  int num_device = device_to_stream.size();
  CHECK(num_device % clique_size == 0);
  device_to_stream.resize(clique_size);
  PartRepMultiStream::Build(stream_id_list, stream_freq_list, device_to_stream, num_node, nid_to_block_tensor);
}
void CliqueLocalFreqSolver::Solve(std::vector<int> device_to_stream,
                         std::vector<int> device_to_cache_percent,
                         std::string mode, double T_local, double T_remote,
                         double T_cpu) {
  int num_device = device_to_stream.size();
  CHECK(num_device % clique_size == 0);
  device_to_stream.resize(clique_size);
  device_to_cache_percent.resize(clique_size);
  PartRepMultiStream::Solve(device_to_stream, device_to_cache_percent, mode, T_local, T_remote, T_cpu);
};
void solve_local_intuitive(TensorPtr stream_id_list, TensorPtr stream_freq_list, const IdType num_node,
    std::vector<int> device_to_stream, std::vector<int> /*device_to_cache_percent*/__,
    TensorPtr nid_to_block, TensorPtr & block_placement,
    std::string _, double T_local, double T_remote, double T_cpu) {
  CHECK_EQ(stream_id_list->Shape().size(), 2);
  CHECK_EQ(stream_freq_list->Shape().size(), 2);
  CHECK_EQ(stream_id_list->Shape(), stream_freq_list->Shape());
  CHECK_EQ(stream_id_list->Shape()[1], num_node);
  IdType num_stream = stream_id_list->Shape()[0];
  IdType num_device = device_to_stream.size();
  auto cpu_ctx = CPU(CPU_CLIB_MALLOC_DEVICE);

  TensorView<IdType> stream_id_list_view(stream_id_list);
  TensorView<IdType> stream_freq_list_view(stream_freq_list);
  /**
   * Map each node to a rank for each stream.
   * Nodes with same rank for every stream forms a block.
   */
  TensorPtr nid_to_local_rank_tensor  = Tensor::Empty(kI32, {num_node, num_stream}, cpu_ctx, "coll_cache.nid_to_local_rank");
  TensorView<uint32_t> nid_to_local_rank(nid_to_local_rank_tensor);
  LOG(INFO) << "mapping nid to rank..." << "num_node=" << num_node << ", num_stream=" << num_stream << ", num_thread=" << RunConfig::omp_thread_num;
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for (uint32_t orig_rank = 0; orig_rank < num_node; orig_rank++) {
    for (uint32_t stream_idx = 0; stream_idx < num_stream; stream_idx++) {
      uint32_t nid = stream_id_list_view[stream_idx][orig_rank].ref();
      nid_to_local_rank[nid][stream_idx].ref() = orig_rank;
    }
  }

  /**
   * aggregate a total freq of all device
   */
  auto gpu_device = Device::Get(GPU(0));
  TensorPtr global_rank_to_freq = Tensor::Empty(kF64, {num_node}, cpu_ctx, "");
  TensorPtr global_rank_to_nid = Tensor::Empty(kI32, {num_node}, cpu_ctx, "");
  {
    double* nid_to_global_freq = gpu_device->AllocArray<double>(GPU(0), num_node);
    double* nid_to_global_freq_out = gpu_device->AllocArray<double>(GPU(0), num_node);
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
    for (IdType nid = 0; nid < num_node; nid++) {
      double sum = 0;
      for (IdType dev_id = 0; dev_id < num_device; dev_id++) {
        IdType stream_id = device_to_stream[dev_id];
        IdType rank = nid_to_local_rank[nid][stream_id].ref();
        sum += stream_freq_list_view[stream_id][rank].ref();
      }
      global_rank_to_freq->Ptr<double>()[nid] = sum;
    }
    gpu_device->CopyDataFromTo(global_rank_to_freq->CPtr<double>(), 0, nid_to_global_freq, 0, global_rank_to_freq->NumBytes(), cpu_ctx, GPU(0));
    gpu_device->SyncDevice(GPU(0));
    IdType* nid = gpu_device->AllocArray<IdType>(GPU(0), num_node);
    IdType* nid_out = gpu_device->AllocArray<IdType>(GPU(0), num_node);
    cuda::ArrangeArray<IdType>(nid, num_node);
    gpu_device->SyncDevice(GPU(0));
    cuda::CubSortPairDescending(nid_to_global_freq, nid_to_global_freq_out, nid, nid_out, num_node, GPU(0));
    gpu_device->SyncDevice(GPU(0));
    gpu_device->CopyDataFromTo(nid_to_global_freq_out, 0, global_rank_to_freq->Ptr<double>(), 0, global_rank_to_freq->NumBytes(), GPU(0), cpu_ctx);
    gpu_device->CopyDataFromTo(nid_out, 0, global_rank_to_nid->Ptr<IdType>(), 0, global_rank_to_nid->NumBytes(), GPU(0), cpu_ctx);
    gpu_device->SyncDevice(GPU(0));
    gpu_device->FreeWorkspace(GPU(0), nid_to_global_freq_out);
    gpu_device->FreeWorkspace(GPU(0), nid_to_global_freq);
    gpu_device->FreeWorkspace(GPU(0), nid_out);
    gpu_device->FreeWorkspace(GPU(0), nid);
  }
  TensorPtr nid_to_global_rank_tensor  = Tensor::Empty(kI32, {num_node}, cpu_ctx, "coll_cache.nid_to_global_rank");
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for (uint32_t global_rank = 0; global_rank < num_node; global_rank++) {
      uint32_t nid = global_rank_to_nid->Ptr<IdType>()[global_rank];
      nid_to_global_rank_tensor->Ptr<IdType>()[nid] = global_rank;
  }

  size_t max_rank = 0;
  for (size_t cache_percent = 0; cache_percent < 100; cache_percent++) {
    size_t rank_start = cache_percent * num_node / 100, rank_stop = (cache_percent + 1) * num_node / 100;
    #pragma omp parallel for num_threads(RunConfig::omp_thread_num) reduction(max : max_rank)
    for (IdType local_rank = rank_start; local_rank < rank_stop; local_rank++) {
      for (IdType stream_id = 0; stream_id < num_stream; stream_id++) {
        IdType nid = stream_id_list_view[stream_id][local_rank].ref();
        IdType global_rank = nid_to_global_rank_tensor->CPtr<IdType>()[nid];
        max_rank = Max<size_t>(max_rank, global_rank);
      }
    }
    std::cout << cache_percent << ":" << max_rank/(double)num_node << "\n";
  }
}
} // namespace coll_cache
}
}