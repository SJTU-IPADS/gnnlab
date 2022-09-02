#include "../common.h"
#include "../run_config.h"
#include "../logging.h"
#include "../device.h"
#include "../cpu/mmap_cpu_device.h"
#include "../cpu/cpu_utils.h"
#include "ndarray.h"
#include <omp.h>
#include <sys/mman.h>
#include <unistd.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <sys/fcntl.h>
#include <cstring>
#include <bitset>
#include "gurobi_c++.h"
#include <tbb/concurrent_unordered_map.h>
#include "optimal_solver_class.h"

namespace samgraph {
namespace common {
namespace coll_cache {

void CollCacheSolver::Solve(std::vector<int> device_to_stream,
                            std::vector<int> device_to_cache_percent,
                            std::string mode, double T_local, double T_cpu) {
  Solve(device_to_stream, device_to_cache_percent, mode, T_local, RunConfig::coll_cache_link_desc.AggregatedRemoteTime(), T_cpu);
};

void OptimalSolver::Build(TensorPtr stream_id_list, TensorPtr stream_freq_list, std::vector<int> device_to_stream, const IdType num_node, const TensorPtr nid_to_block_tensor) {
  CHECK_EQ(stream_id_list->Shape().size(), 2);
  CHECK_EQ(stream_freq_list->Shape().size(), 2);
  CHECK_EQ(stream_id_list->Shape(), stream_freq_list->Shape());
  CHECK_EQ(stream_id_list->Shape()[1], num_node);
  IdType num_stream = stream_id_list->Shape()[0];
  auto cpu_ctx = CPU(CPU_CLIB_MALLOC_DEVICE);
  // coarse-grained slice to reduce asymm solve time
  // symmetric & switch's precision still holds
  max_size_per_block = num_node / 1000;
  auto freq_to_slot = [this](float freq, uint32_t rank, IdType num_node){ return this->freq_to_slot_1(freq, rank, num_node);};

  TensorPtr nid_to_rank_tensor  = Tensor::Empty(kI32, {num_node, num_stream}, cpu_ctx, "coll_cache.nid_to_rank");
  TensorPtr nid_to_slot_tensor  = Tensor::Empty(kI32, {num_node, num_stream}, cpu_ctx, "coll_cache.nid_to_slot");
  // nid_to_block_tensor = Tensor::Empty(kI32, {num_node}, cpu_ctx, "coll_cache.nid_to_block");
  CHECK(nid_to_block_tensor->Shape() == std::vector<size_t>{num_node});

  TensorView<uint32_t> nid_to_rank(nid_to_rank_tensor);
  TensorView<uint32_t> nid_to_slot(nid_to_slot_tensor);
  TensorView<uint32_t> nid_to_block(nid_to_block_tensor);

  concurrent_full_slot_map slot_array_to_full_block;

  TensorView<IdType> stream_id_list_view(stream_id_list);
  TensorView<IdType> stream_freq_list_view(stream_freq_list);

  // identify freq boundary of first slot
  for (IdType i = 0; i < num_stream; i++) {
    if (alpha < stream_freq_list_view[i][(num_node - 1) / 100].ref()) {
      alpha = stream_freq_list_view[i][(num_node - 1) / 100].ref();
    }
  }

  /**
   * Map each node to a rank for each stream.
   * Nodes with same rank for every stream forms a block.
   */
  LOG(WARNING) << "mapping nid to rank...";
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for (uint32_t orig_rank = 0; orig_rank < num_node; orig_rank++) {
    for (uint32_t stream_idx = 0; stream_idx < num_stream; stream_idx++) {
      uint32_t nid = stream_id_list_view[stream_idx][orig_rank].ref();
      nid_to_rank[nid][stream_idx].ref() = orig_rank;
    }
  }
  auto slot_list_to_full_block_id = [num_stream](const TensorView<uint32_t>& slot_array){
    CHECK(*slot_array._shape == num_stream);
    size_t ret = 0;
    for (size_t i = 0; i < num_stream; i++) {
      ret *= num_stream;
      ret += slot_array._data[i];
    }
    return ret;
  };
  LOG(WARNING) << "counting slots...";
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for (uint32_t nid = 0; nid < num_node; nid++) {
    // for each nid, prepare a slot list
    for (uint32_t stream_idx = 0; stream_idx < num_stream; stream_idx++) {
      uint32_t orig_rank = nid_to_rank[nid][stream_idx].ref();
      double freq = stream_freq_list_view[stream_idx][orig_rank].ref();
      int slot_id = freq_to_slot(freq, orig_rank, num_node);
      nid_to_slot[nid][stream_idx].ref() = slot_id;
    }
    // map the slot list to block
    size_t seq_id = slot_list_to_full_block_id(nid_to_slot[nid]);
    nid_to_block[nid].ref() = slot_array_to_full_block.register_bucket(seq_id);
  }
  LOG(WARNING) << "Final num slot is " << slot_array_to_full_block.next_free_slot.load();
  block_identifer* buckets = new block_identifer[slot_array_to_full_block.next_free_slot.load()];
  next_free_block.store(0);

  LOG(WARNING) << "counting blocks...";
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for (uint32_t nid = 0; nid < num_node; nid++) {
    nid_to_block[nid].ref() = buckets[nid_to_block[nid].ref()].add_node(this);
  }
  delete[] buckets;

  uint32_t total_num_blocks = next_free_block.load();
  LOG(WARNING) << "Final num block is " << total_num_blocks;

  /**
   * Sum frequency & density of each block
   */
  LOG(WARNING) << "counting freq and density...";
  block_density_tensor = Tensor::Empty(kF64, {total_num_blocks}, cpu_ctx, "coll_cache.block_density_tensor");
  block_freq_tensor    = Tensor::Empty(kF64, {total_num_blocks, num_stream}, cpu_ctx, "coll_cache.block_freq_tensor");

  std::memset(block_density_tensor->MutableData(), 0, block_density_tensor->NumBytes());
  std::memset(block_freq_tensor->MutableData(), 0, block_freq_tensor->NumBytes());

  TensorView<double> block_density_array(block_density_tensor);
  TensorView<double> block_freq_array(block_freq_tensor);
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for (int thread_idx = 0; thread_idx < RunConfig::omp_thread_num; thread_idx++) {
    for (uint32_t nid = 0; nid < num_node; nid++) {
      uint32_t block_id = nid_to_block[nid].ref();
      if (std::hash<uint64_t>()(block_id) % RunConfig::omp_thread_num != thread_idx) {
        continue;
      }
      block_density_array[block_id].ref() += 1;
      for (uint32_t stream_idx = 0; stream_idx < num_stream; stream_idx++) {
        uint32_t orig_rank = nid_to_rank[nid][stream_idx].ref();
        double freq = stream_freq_list_view[stream_idx][orig_rank].ref();
        // assign all zero freq a minimal freq to handle touched node << cache space
        freq = std::max(freq, 1e-3);
        block_freq_array[block_id][stream_idx].ref() += freq;
      }
    }
  }

  /**
   * Average the frequency for each block
   */
  LOG(WARNING) << "averaging freq and density...";
  LOG(WARNING) << block_density_tensor->NumItem();
// #pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for (uint32_t block_id = 0; block_id < block_density_tensor->NumItem(); block_id++) {
    if (block_density_array[block_id].ref() == 0) continue; 
    for (uint32_t stream_id = 0; stream_id < num_stream; stream_id++) {
      block_freq_array[{block_id,stream_id}].ref() /= block_density_array[block_id].ref() ;
    }
    block_density_array[block_id].ref() *= 100/(double)num_node ;
    // std::cout << block_density_array[block_id].ref() << " ";
  }
  // std::cout << "\n";
  block_placement = Tensor::CreateShm(Constant::kCollCachePlacementShmName, kU8, block_density_tensor->Shape(), "coll_cache_block_placement");
}

void OptimalSolver::Solve(std::vector<int> device_to_stream, std::vector<int> device_to_cache_percent, std::string mode, double T_local, double T_remote, double T_cpu) {
  CHECK(RunConfig::coll_cache_link_desc._topo_type != AsymmLinkDesc::kHardWiredAsymm) << "OptimalSolver does not support asymm link topo";
  CHECK(mode == "BIN");
  CHECK(block_density_tensor->Defined());
  double* block_density_array = block_density_tensor->Ptr<double>();
  TensorView<double> block_freq_array(block_freq_tensor);
  int cache_percent   = device_to_cache_percent.at(0);
  uint32_t num_device = device_to_stream.size();
  IdType num_stream   = block_freq_tensor->Shape().at(1);
  uint32_t num_block  = block_density_tensor->Shape().at(0);
  LOG(INFO) << "constructing optimal solver, device=" << num_device << ", stream=" << num_stream;

  std::cerr << num_block << " blocks, " << num_device << " devices\n";

  GRBEnv env = GRBEnv(true);
  env.set("LogFile", "cppsolver.log");
  // env.set(GRB_IntParam_Presolve, 0);
  // // env.set(GRB_IntParam_Method, 2);
  // // env.set(GRB_IntParam_Method, -1);
  // env.set(GRB_IntParam_Method, 3);
  // env.set(GRB_IntParam_Threads, RunConfig::omp_thread_num);
  // env.set(GRB_DoubleParam_BarConvTol, 1e-3);
  // env.set(GRB_DoubleParam_OptimalityTol, 1e-2);
  // env.set(GRB_DoubleParam_MIPGap, 2e-3);


  env.set(GRB_IntParam_ConcurrentMIP, 50);
  // env.set(GRB_DoubleParam_Heuristics, 0.5);
  // env.set(GRB_DoubleParam_NoRelHeurTime, 10);
  env.set(GRB_IntParam_Presolve, 1);
  env.set(GRB_IntParam_MIPFocus, 1);
  // env.set(GRB_IntParam_Crossover, 0);
  // env.set(GRB_IntParam_Presolve, -1);
  env.set(GRB_IntParam_Method, 2);
  // env.set(GRB_IntParam_NodeMethod, 2);
  // env.set(GRB_IntParam_Method, -1);
  // env.set(GRB_IntParam_Method, 3);
  env.set(GRB_IntParam_Threads, RunConfig::omp_thread_num*2);
  env.set(GRB_DoubleParam_BarConvTol, 1e-4);
  env.set(GRB_DoubleParam_OptimalityTol, 2e-2);
  env.set(GRB_DoubleParam_MIPGap, 2e-2);

  env.start();

  GRBModel model = GRBModel(env);

  GRBVar z = model.addVar(0.0, std::numeric_limits<double>::max(), 0.0, GRB_CONTINUOUS, "z");
  TensorPtr c_list_tensor = Tensor::Empty(kI64, {num_block}, CPU(CPU_CLIB_MALLOC_DEVICE), "c_list");
  TensorPtr x_list_tensor = Tensor::Empty(kI64, {num_block, num_device}, CPU(CPU_CLIB_MALLOC_DEVICE), "x_list");

  TensorView<GRBVar> c_list(c_list_tensor);
  TensorView<GRBVar> x_list(x_list_tensor);
  std::vector<GRBLinExpr> time_list(num_device);
  std::vector<GRBLinExpr> local_weight_list(num_device);
  std::vector<double>     total_weight_list(num_device);
  std::vector<GRBLinExpr> cpu_weight_list(num_device);

  auto constraint_connect_c_x = [&](GRBModel & model, uint32_t block_id) {
    if (block_density_array[block_id] == 0) return;
    GRBLinExpr expr;
    expr += c_list[block_id].ref();
    FOR_LOOP(device_id, num_device) {
      expr += x_list[block_id][device_id].ref();
    }
    model.addConstr(expr >= 1);
    
    if (mode == "BIN") {
      GRBLinExpr expr;
      expr += c_list[block_id].ref() * num_device;
      FOR_LOOP(device_id, num_device) {
        expr += x_list[block_id][device_id].ref();
      }
      model.addConstr(expr <= num_device);
    }
  };
  auto  constraint_connect_r_x = [&](GRBModel & model, uint32_t block_id, uint32_t device_id) {
    if (mode == "CONT") {
      if (block_density_array[block_id] == 0) return;
      model.addConstr(c_list[block_id].ref() + x_list[block_id][device_id].ref() <= 1);
    }
  };
  auto constraint_capacity = [&](GRBModel & model, uint32_t device_id) {
    GRBLinExpr expr;
    FOR_LOOP(block_id, num_block) {
      if (block_density_array[block_id] == 0) continue;
      expr += x_list[block_id][device_id].ref() * block_density_array[block_id];
    }
    model.addConstr(expr <= cache_percent);
  };
  auto constraint_time = [&](GRBModel & model, uint32_t device_id) {
    uint32_t stream_id = device_to_stream[device_id];
    double sum_weight = 0;
    GRBLinExpr &expr = time_list[device_id];
    FOR_LOOP(block_id, num_block) {
      double weight = block_density_array[block_id] * block_freq_array[block_id][stream_id].ref();
      if (weight == 0) continue;
      sum_weight += weight;
      expr += c_list[block_id].ref() * (weight * (T_cpu - T_remote)) - x_list[block_id][device_id].ref() * (weight * (T_remote - T_local));

      local_weight_list[device_id] +=  weight * x_list[block_id][device_id].ref();
      cpu_weight_list[device_id]   +=  weight * c_list[block_id].ref();
    }
    expr += sum_weight * T_remote;
    total_weight_list[device_id] = sum_weight;
    model.addConstr(expr <= z, "time_" + std::to_string(device_id));
  };

  LOG(INFO) << "Add Var...";
  char var_type = (mode == "BIN") ? GRB_BINARY : GRB_CONTINUOUS;
  FOR_LOOP(block_id, num_block) {
    if (ignore_block(block_id, block_density_array[block_id])) {
      continue;
    }
    c_list[block_id].ref() = model.addVar(0, 1, 0, var_type);
    FOR_LOOP(device_id, num_device) {
      x_list[block_id][device_id].ref() = model.addVar(0, 1, 0, var_type);
    }
  }

  LOG(INFO) << "Capacity...";
  FOR_LOOP(device_id, num_device) {constraint_capacity(model, device_id);}

  LOG(INFO) << "Connect CPU...";
  FOR_LOOP(block_id, num_block) {constraint_connect_c_x(model, block_id);}

  LOG(INFO) << "Connect Remote...";
  FOR_LOOP(block_id, num_block) {
    FOR_LOOP(device_id, num_device) {constraint_connect_r_x(model, block_id, device_id);}
  }

  LOG(INFO) << "Time...";
  FOR_LOOP(device_id, num_device) {constraint_time(model, device_id);}

  model.setObjective(z + 0, GRB_MINIMIZE);

  model.optimize();

  CHECK(num_device <= 8);
  CHECK(block_placement->Shape() == std::vector<size_t>{num_block});
  TensorView<uint8_t> block_placement_array(block_placement);
  LOG(INFO) << "Coll Cache init block placement array";
  // #pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  FOR_LOOP(block_id, num_block) {
    block_placement_array[block_id].ref() = 0;
    // std::ios_base::fmtflags f( std::cerr.flags() );
    // std::cerr << "block " << block_id
    //           << std::fixed << std::setw(8) << std::setprecision(6)
    //           << ", density=" << block_density_array[block_id]
    //           << std::fixed << std::setw(8) << std::setprecision(3)
    //           << ", freq=" << block_freq_array[block_id][0].ref();
    if (!ignore_block(block_id, block_density_array[block_id])) {
      FOR_LOOP(device_id, num_device) {
        uint8_t x_result = (uint8_t)std::round(x_list[block_id][device_id].ref().get(GRB_DoubleAttr::GRB_DoubleAttr_X));
        block_placement_array[block_id].ref() |= (x_result << device_id);
      }
    }
    std::bitset<8> bs(block_placement_array[block_id].ref());
    // std::cerr << "  storage is " << bs << "\n";
    // std::cerr.flags(f);
  }
  std::cout << "coll_cache:optimal_local_rate=";
  FOR_LOOP(part_id, num_device) { std::cout << local_weight_list[part_id].getValue() / total_weight_list[part_id] << ","; }
  std::cout << "\n";
  std::cout << "coll_cache:optimal_remote_rate=";
  FOR_LOOP(part_id, num_device) { std::cout << 1 - (local_weight_list[part_id].getValue() + cpu_weight_list[part_id].getValue()) / total_weight_list[part_id] << ","; }
  std::cout << "\n";
  std::cout << "coll_cache:optimal_cpu_rate=";
  FOR_LOOP(part_id, num_device) { std::cout << cpu_weight_list[part_id].getValue() / total_weight_list[part_id] << ","; }
  std::cout << "\n";
  std::cout << "z=" << z.get(GRB_DoubleAttr::GRB_DoubleAttr_X) << "\n";
  LOG(INFO) << "Coll Cache init block placement array done";
  model.reset(1);
  LOG(INFO) << "Coll Cache model reset done";
}

void OptimalAsymmLinkSolver::Solve(std::vector<int> device_to_stream, std::vector<int> device_to_cache_percent, std::string mode, double T_local, double T_cpu) {
  CHECK(mode == "BIN");
  CHECK(block_density_tensor->Defined());
  double*            block_density_array = block_density_tensor->Ptr<double>();
  TensorView<double> block_freq_array(block_freq_tensor);
  int      cache_percent  = device_to_cache_percent.at(0);
  uint32_t num_device     = device_to_stream.size();
  IdType   num_stream     = block_freq_tensor->Shape().at(1);
  uint32_t num_block      = block_density_tensor->Shape().at(0);
  uint32_t num_link       = link_src[0].size(); // we simply assume all dst device have same num link
  LOG(WARNING) << "constructing optimal solver, device=" << num_device << ", stream=" << num_stream;

  std::cerr << num_block << " blocks, " << num_device << " devices\n";

  GRBEnv env = GRBEnv(true);
  env.set("LogFile", "cppsolver.log");
  env.set(GRB_IntParam_ConcurrentMIP, 50);
  // env.set(GRB_DoubleParam_Heuristics, 0.5);
  // env.set(GRB_DoubleParam_NoRelHeurTime, 10);
  env.set(GRB_IntParam_Presolve, 1);
  env.set(GRB_IntParam_MIPFocus, 1);
  // env.set(GRB_IntParam_Crossover, 0);
  // env.set(GRB_IntParam_Presolve, -1);
  // env.set(GRB_IntParam_Method, 2);
  // env.set(GRB_IntParam_NodeMethod, 2);
  // env.set(GRB_IntParam_Method, -1);
  env.set(GRB_IntParam_Method, 3);
  env.set(GRB_IntParam_Threads, RunConfig::omp_thread_num*2);
  env.set(GRB_DoubleParam_BarConvTol, 1e-4);
  env.set(GRB_DoubleParam_OptimalityTol, 1e-2);
  env.set(GRB_DoubleParam_MIPGap, 1e-2);

  env.set(GRB_DoubleParam_TimeLimit, 200);

  // tuned
  env.set(GRB_DoubleParam_MIPGap, 0.03);
  env.set(GRB_IntParam_MIPFocus, 2);
  env.set(GRB_IntParam_ConcurrentMIP, 36);
  env.set(GRB_IntParam_Threads, 108);
  env.set(GRB_IntParam_BranchDir, 1);
  env.set(GRB_IntParam_AggFill, 100);
  env.set(GRB_IntParam_NormAdjust, 3);
  env.set(GRB_IntParam_Presolve, 2);
  env.set(GRB_IntParam_SimplexPricing, 2);
  env.set(GRB_IntParam_DegenMoves, 0);
  env.set(GRB_IntParam_CutPasses, 5);
  env.set(GRB_IntParam_PrePasses, 8);
  env.set(GRB_DoubleParam_Heuristics, 0.001);
  env.set(GRB_IntParam_ScaleFlag, 0);
  env.set(GRB_IntParam_StrongCGCuts, 0);
  env.set(GRB_IntParam_MIRCuts, 1);
  env.set(GRB_IntParam_Cuts, 3);

  env.start();

  GRBModel model = GRBModel(env);

  /**
   * z: final goal, max time of each dst gpu, minimize it
   * x: storage of each block on each src gpu
   * a: access flag of each block: a[block][dst][src] means whether dst read this block from src
   */
  GRBVar z = model.addVar(0.0, std::numeric_limits<double>::max(), 0.0, GRB_CONTINUOUS, "z");
  TensorPtr c_list_tensor = Tensor::Empty(kI64, {num_block}, CPU(CPU_CLIB_MALLOC_DEVICE), "c_list");
  TensorPtr x_list_tensor = Tensor::Empty(kI64, {num_block, num_device}, CPU(CPU_CLIB_MALLOC_DEVICE), "x_list");
  TensorPtr a_list_tensor = Tensor::Empty(kI64, {num_block, num_device, num_link}, CPU(CPU_CLIB_MALLOC_DEVICE), "a_list");

  TensorView<GRBVar> c_list(c_list_tensor);
  TensorView<GRBVar> x_list(x_list_tensor);
  TensorView<GRBVar> a_list(a_list_tensor);

  // std::vector<GRBLinExpr> time_list(num_device);
  std::vector<GRBLinExpr> local_cpu_time_list(num_device);
  vec<vec<GRBLinExpr>>    remote_time_list(num_device, vec<GRBLinExpr>(num_link));
  std::vector<GRBLinExpr> local_weight_list(num_device);
  std::vector<double>     total_weight_list(num_device);
  std::vector<GRBLinExpr> cpu_weight_list(num_device);

  // for each dst, 
  // sum a_dst_src <= 1
  // src
  auto constraint_connect_a_c = [&](GRBModel & model, uint32_t block_id) {
    if (block_density_array[block_id] == 0) return;
    FOR_LOOP(dst_dev, num_device) {
      GRBLinExpr expr;
      FOR_LOOP(src_link, num_link) {
        expr += a_list[block_id][dst_dev][src_link].ref();
      }
      model.addConstr(expr + c_list[block_id].ref() + x_list[block_id][dst_dev].ref() == 1);
    }
  };

  // for each src,
  // x_src >= max(a_dst_src)
  //          dst
  auto constraint_connect_a_x = [&](GRBModel & model, uint32_t block_id) {
    if (block_density_array[block_id] == 0) return;
    FOR_LOOP(dst_dev, num_device) {
      FOR_LOOP(src_link, num_link) {
        GRBLinExpr expr;
        for (auto src_dev : link_src[dst_dev][src_link]) {
          expr += x_list[block_id][src_dev].ref();
        }
        model.addConstr(expr >= a_list[block_id][dst_dev][src_link].ref());
      }
    }
  };

  // for each src,
  //  sum x_src < cache size
  // block
  auto constraint_capacity = [&](GRBModel & model, uint32_t src_dev) {
    GRBLinExpr expr;
    FOR_LOOP(block_id, num_block) {
      if (block_density_array[block_id] == 0) continue;
      expr += x_list[block_id][src_dev].ref() * block_density_array[block_id];
    }
    model.addConstr(expr <= cache_percent);
  };
  auto constraint_time = [&](GRBModel & model, uint32_t dst_dev) {
    uint32_t stream_id = device_to_stream[dst_dev];
    double sum_weight = 0;
    GRBLinExpr &local_cpu_time = local_cpu_time_list[dst_dev];
    FOR_LOOP(block_id, num_block) {
      double weight = block_density_array[block_id] * block_freq_array[block_id][stream_id].ref();
      if (weight == 0) continue;
      sum_weight += weight;
      local_cpu_time += c_list[block_id].ref() * T_cpu * weight;
      local_cpu_time += x_list[block_id][dst_dev].ref() * T_local * weight;

      local_weight_list[dst_dev] +=  weight * x_list[block_id][dst_dev].ref();
      cpu_weight_list[dst_dev]   +=  weight * c_list[block_id].ref();
    }
    FOR_LOOP(src_link, num_link) {
      GRBLinExpr &remote_time = remote_time_list[dst_dev][src_link];
      FOR_LOOP(block_id, num_block) {
        double weight = block_density_array[block_id] * block_freq_array[block_id][stream_id].ref();
        if (weight == 0) continue;
        remote_time += a_list[block_id][dst_dev][src_link].ref() * link_time[dst_dev][src_link] * weight;
      }
      model.addConstr(remote_time + local_cpu_time <= z);
    }
    total_weight_list[dst_dev] = sum_weight;
  };

  LOG(WARNING) << "Add Var...";
  char var_type = (mode == "BIN") ? GRB_BINARY : GRB_CONTINUOUS;
  FOR_LOOP(block_id, num_block) {
    if (ignore_block(block_id, block_density_array[block_id])) {
      continue;
    }
    c_list[block_id].ref() = model.addVar(0, 1, 0, var_type);
    FOR_LOOP(dst_dev, num_device) {
      x_list[block_id][dst_dev].ref() = model.addVar(0, 1, 0, var_type);
      FOR_LOOP(src_link, num_link) {
        a_list[block_id][dst_dev][src_link].ref() = model.addVar(0, 1, 0, var_type);
      }
    }
  }

  LOG(WARNING) << "Capacity...";
  FOR_LOOP(device_id, num_device) {constraint_capacity(model, device_id);}

  LOG(WARNING) << "Connect CPU...";
  FOR_LOOP(block_id, num_block) {constraint_connect_a_c(model, block_id);}
  LOG(WARNING) << "Connect Access To Storage...";
  FOR_LOOP(block_id, num_block) {
    FOR_LOOP(device_id, num_device) {constraint_connect_a_x(model, block_id);}
  }
  LOG(WARNING) << "Time...";
  FOR_LOOP(device_id, num_device) {constraint_time(model, device_id);}

  model.setObjective(z + 0, GRB_MINIMIZE);

  model.write("asymm.lp");

  model.optimize();

  CHECK(num_device <= 8);
  CHECK(block_placement->Shape() == std::vector<size_t>{num_block});
  // num_link + local + cpu always <= 8
  block_access_from = Tensor::CreateShm(Constant::kCollCacheAccessShmName, kU8, {num_device, num_block}, "coll_cache_block_access");
  TensorView<uint8_t> block_placement_array(block_placement);
  TensorView<uint8_t> block_access_from_array(block_access_from);
  LOG(WARNING) << "Coll Cache init block placement array";
  // #pragma omp parallel for num_threads(RunConfig::omp_thread_num)

  auto get_val = [](GRBVar & var) { return var.get(GRB_DoubleAttr::GRB_DoubleAttr_X);};
  auto get_int = [](GRBVar & var) { return std::round(var.get(GRB_DoubleAttr::GRB_DoubleAttr_X));};

  vec<vec<uint8_t>> link_bitmap(num_device, vec<uint8_t>(num_link, 0));
  for (uint32_t dev_id = 0; dev_id < num_device; dev_id++) {
    for (uint32_t link = 0; link < num_link; link++) {
      uint8_t & bitmap = link_bitmap[dev_id][link];
      for (auto src_dev : this->link_src[dev_id][link]) {
        bitmap |= 1 << src_dev;
      }
    }
  }

  vec<uint8_t> bitmap_to_src_dev(1 << num_device, 0);
  PreDecideSrc(num_device, num_device + 1, bitmap_to_src_dev.data());

  FOR_LOOP(block_id, num_block) {
    // by default, this block is placed at cpu
    block_placement_array[block_id].ref() = 0;
    FOR_LOOP(device_id, num_device) {
      block_access_from_array[device_id][block_id].ref() = num_device; // num_device is treat as cpu
    }

    // std::ios_base::fmtflags f( std::cerr.flags() );
    // std::cerr << "block " << block_id
    //           << std::fixed << std::setw(8) << std::setprecision(6)
    //           << ", density=" << block_density_array[block_id]
    //           << std::fixed << std::setw(8) << std::setprecision(3)
    //           << ", freq=" << block_freq_array[block_id][0].ref();

    // x == 1 -> access from = local dev id
    // x == 0, a != 0 -> <bitmap of this link> & <storage bitmap>, then choose one from it
    // x == 0, a == 0 -> cpu
    if (ignore_block(block_id, block_density_array[block_id])) {
      continue;
    }
    FOR_LOOP(device_id, num_device) {
      uint8_t x_result = (uint8_t)std::round(x_list[block_id][device_id].ref().get(GRB_DoubleAttr::GRB_DoubleAttr_X));
      block_placement_array[block_id].ref() |= (x_result << device_id);
    }
    // build access from
    FOR_LOOP(device_id, num_device) {
      if (get_int(x_list[block_id][device_id].ref())) {
        block_access_from_array[device_id][block_id].ref() = device_id;
        continue;
      }
      if (get_int(c_list[block_id].ref())) {
        block_access_from_array[device_id][block_id].ref() = num_device; // num_device is treat as cpu
        continue;
      }
      for (uint32_t src_link = 0; src_link < num_link; src_link ++) {
        if (get_int(a_list[block_id][device_id][src_link].ref()) == 0) continue;
        const uint8_t link_src_bitmap = link_bitmap[device_id][src_link];
        const uint8_t storage_bit_map = block_placement_array[block_id].ref();
        const uint8_t candidate_bit_map = link_src_bitmap & storage_bit_map;
        block_access_from_array[device_id][block_id].ref() = bitmap_to_src_dev[candidate_bit_map];
        break;
      }
    }
    // std::bitset<8> bs(block_placement_array[block_id].ref());
    // std::cerr << "  storage is " << bs << "\n";
    // std::cerr.flags(f);
  }
  std::cout << "coll_cache:optimal_local_rate=";
  FOR_LOOP(part_id, num_device) { std::cout << local_weight_list[part_id].getValue() / total_weight_list[part_id] << ","; }
  std::cout << "\n";
  std::cout << "coll_cache:optimal_remote_rate=";
  FOR_LOOP(part_id, num_device) { std::cout << 1 - (local_weight_list[part_id].getValue() + cpu_weight_list[part_id].getValue()) / total_weight_list[part_id] << ","; }
  std::cout << "\n";
  std::cout << "coll_cache:optimal_cpu_rate=";
  FOR_LOOP(part_id, num_device) { std::cout << cpu_weight_list[part_id].getValue() / total_weight_list[part_id] << ","; }
  std::cout << "\n";
  std::cout << "z=" << z.get(GRB_DoubleAttr::GRB_DoubleAttr_X) << "\n";
  LOG(WARNING) << "Coll Cache init block placement array done";
  model.reset(1);
  LOG(WARNING) << "Coll Cache model reset done";
}
void OptimalAsymmLinkSolver::PreDecideSrc(int num_bits,
                                          int cpu_location_id,
                                          uint8_t *placement_to_src) {
  auto g = std::default_random_engine(
      std::chrono::system_clock::now().time_since_epoch().count());
  for (int placement = 0; placement < (1 << num_bits); placement++) {
    if (placement == 0) {
      placement_to_src[placement] = cpu_location_id;
    } else {
      int num_nz = cpu::CountBits(placement);
      int location = 0;
      // scan all 1, and choose current 1 uniformly
      for (; location < num_bits; location++) {
        if ((placement & (1 << location)) == 0) continue;
        int choice = std::uniform_int_distribution<int>(1, num_nz)(g);
        if (choice == 1) {
          break;
        }
        num_nz--;
      }
      placement_to_src[placement] = location;
    }
  }
}

void SingleStreamSolverBase::Build(TensorPtr stream_id_list,
                                   TensorPtr stream_freq_list,
                                   std::vector<int> device_to_stream,
                                   const IdType num_node,
                                   const TensorPtr nid_to_block_tensor) {
  this->stream_id_list = stream_id_list;
  this->stream_freq_list = stream_freq_list;
  this->nid_to_block = nid_to_block_tensor;
  CHECK(stream_id_list->Shape()[0] == 1);
  CHECK(stream_freq_list->Shape()[0] == 1);
}
void IntuitiveSolver::Solve(std::vector<int> device_to_stream,
                            std::vector<int> device_to_cache_percent,
                            std::string mode, double T_local, double T_remote,
                            double T_cpu) {
  CHECK(RunConfig::coll_cache_link_desc._topo_type != AsymmLinkDesc::kHardWiredAsymm) << "IntuitiveSolver does not support asymm link topo";
  CHECK(std::accumulate(device_to_stream.begin(), device_to_stream.end(), 0, std::plus<>()) == 0);
  const int num_device = device_to_stream.size();
  const int num_block = num_device + 2;
  const IdType num_node = stream_freq_list->Shape()[1];
  // the freq list must already be sorted
  // now calculate the boundary
  const IdType num_cached_nodes = num_node * (device_to_cache_percent[0] / (double)100);
  IdType partition_size_min = 0;
  IdType partition_size_max = (num_device == 1) ? 0 : std::min(num_cached_nodes, (num_node - num_cached_nodes) / (num_device - 1));

  LOG(ERROR) << "num_cached_nodes = " << num_cached_nodes;
  LOG(ERROR) << "[" << partition_size_min << "," << partition_size_max << "]";

  // IdType partition_size = 0;

  auto partition_lb = [&](IdType partition_size) {
    return num_cached_nodes - partition_size;
  };
  auto partition_rb = [&](IdType partition_size) {
    return std::max<IdType>(num_cached_nodes + partition_size * (num_device - 1), 1) - 1;
  };

  // const double T_partition = (T_local + (num_device - 1) * T_remote) / num_device;
  const double mu = 1 + (T_cpu - T_remote) / (T_remote - T_local) * num_device;
  const IdType * freq_array = stream_freq_list->Ptr<IdType>();

  LOG(ERROR) << "mu = " << mu;

  /**
   * |   replicate  |   p    |  p * (n_d-1) | cpu |
   *        ^           ^    ^
   *       >mu     mu  <mu   1
   *       max             min
   */

  IdType partition_size = partition_size_min;

  if (mu < 1) {
    // the best choice is to replicate as much as possible. no partition
    partition_size = partition_size_min;
  } else if (freq_array[partition_lb(partition_size_max)] < freq_array[partition_rb(partition_size_max)] * mu) {
    // we have to choose largest partition
    // build the mapping
    partition_size = partition_size_max;
  } else {
    // now we need to iterate from min to max to find the mu
    while (partition_size_max - partition_size_min > 1) {
      partition_size = (partition_size_min + partition_size_max) / 2;
      if (freq_array[partition_lb(partition_size)] < freq_array[partition_rb(partition_size)] * mu) {
        // go left
        partition_size_min = partition_size;
      } else {
        // go right
        partition_size_max = partition_size;
      }
      LOG(DEBUG) << "[" << partition_size_min << "," << partition_size_max << "]";
    }
    partition_size = partition_size_max;
  }
  
  double rep_w = 0, cpu_w = 0, total_w = 0;
  IdType rep_d = 0, cpu_d = 0;

#pragma omp parallel for num_threads(RunConfig::omp_thread_num) reduction(+ : rep_w, cpu_w, total_w, rep_d, cpu_d)
  for (IdType rank = 0; rank < num_node; rank++) {
    IdType node_id = stream_id_list->Ptr<IdType>()[rank];
    if (rank < partition_lb(partition_size)) {
      // replicate this
      nid_to_block->Ptr<IdType>()[node_id] = 0;
      rep_w += freq_array[rank];
      rep_d ++;
    } else if (rank <= partition_rb(partition_size)) {
      nid_to_block->Ptr<IdType>()[node_id] = (rank % num_device) + 1;
    } else {
      nid_to_block->Ptr<IdType>()[node_id] = num_device + 1;
      cpu_w += freq_array[rank];
      cpu_d++;
    }
    total_w += freq_array[rank];
  }
  double local_w = rep_w + (total_w - cpu_w - rep_w) / num_device;
  double remote_w = (total_w - cpu_w - rep_w) / num_device * (num_device - 1);
  std::cout << "coll_cache:optimal_rep_storage=" << partition_lb(partition_size) / (double)num_node << "\n";
  std::cout << "coll_cache:optimal_part_storage=" << (partition_rb(partition_size) - partition_lb(partition_size)) / (double)num_node << "\n";
  std::cout << "coll_cache:optimal_cpu_storage=" << 1 - (partition_rb(partition_size) / (double)num_node) << "\n";
  std::cout << "coll_cache:optimal_local_storage=" << (partition_lb(partition_size) + partition_size) / (double)num_node << "\n";
  std::cout << "coll_cache:optimal_remote_storage=" << partition_size * (num_device - 1) / (double)num_node << "\n";
  std::cout << "coll_cache:optimal_local_rate=" << local_w / total_w << "\n";
  std::cout << "coll_cache:optimal_remote_rate=" << remote_w / total_w << "\n";
  std::cout << "coll_cache:optimal_cpu_rate=" << cpu_w / total_w << "\n";
  std::cout << "z=" << local_w * 100 / num_node * T_local + remote_w * 100 / num_node * T_remote + cpu_w * 100 / num_node * T_cpu << "\n";

  block_placement = Tensor::CreateShm(Constant::kCollCachePlacementShmName, kU8, {static_cast<size_t>(num_block)}, "coll_cache_block_placement");

  block_placement->Ptr<uint8_t>()[0] = (1 << num_device) - 1;
  for (int i = 0; i < num_device; i++) {
    block_placement->Ptr<uint8_t>()[i + 1] = (1 << i);
  }
  block_placement->Ptr<uint8_t>()[num_device + 1] = 0;
}

void PartitionSolver::Solve(std::vector<int> device_to_stream,
                            std::vector<int> device_to_cache_percent,
                            std::string mode, double T_local, double T_remote,
                            double T_cpu) {
  CHECK(RunConfig::coll_cache_link_desc._topo_type != AsymmLinkDesc::kHardWiredAsymm) << "PartitionSolver does not support asymm link topo";
  CHECK(stream_id_list->Shape()[0] == 1);
  CHECK(stream_freq_list->Shape()[0] == 1);
  CHECK(std::accumulate(device_to_stream.begin(), device_to_stream.end(), 0, std::plus<>()) == 0);
  const int num_device = device_to_stream.size();
  const int num_block = num_device + 1;
  const IdType num_node = stream_freq_list->Shape()[1];
  // the freq list must already be sorted
  // now calculate the boundary
  const IdType num_cached_nodes = num_node * (device_to_cache_percent[0] / (double)100);

  // LOG(ERROR) << "num_cached_nodes = " << num_cached_nodes;
  CHECK_EQ(stream_freq_list->Type(), kI32);
  const IdType * freq_array = stream_freq_list->Ptr<IdType>();

  const IdType partition_size = std::min(num_cached_nodes, num_node / num_device);

  double cpu_w = 0, total_w = 0;

#pragma omp parallel for num_threads(RunConfig::omp_thread_num) reduction(+ : cpu_w, total_w)
  for (IdType rank = 0; rank < num_node; rank++) {
    IdType node_id = stream_id_list->Ptr<IdType>()[rank];
    if (rank < partition_size * num_device) {
      nid_to_block->Ptr<IdType>()[node_id] = rank % num_device;
    } else {
      nid_to_block->Ptr<IdType>()[node_id] = num_device;
      cpu_w += freq_array[rank];
    }
    total_w += freq_array[rank];
  }
  double local_w = (total_w - cpu_w) / num_device;
  double remote_w = (total_w - cpu_w) / num_device * (num_device - 1);
  std::cout << "coll_cache:optimal_rep_storage=" << 0 << "\n";
  std::cout << "coll_cache:optimal_part_storage=" << partition_size * num_device / (double)num_node << "\n";
  std::cout << "coll_cache:optimal_cpu_storage=" << 1 - (partition_size * num_device / (double)num_node) << "\n";
  std::cout << "coll_cache:optimal_local_storage=" << partition_size / (double)num_node << "\n";
  std::cout << "coll_cache:optimal_remote_storage=" << partition_size * (num_device - 1) / (double)num_node << "\n";
  std::cout << "coll_cache:optimal_local_rate=" << local_w / total_w << "\n";
  std::cout << "coll_cache:optimal_remote_rate=" << remote_w / total_w << "\n";
  std::cout << "coll_cache:optimal_cpu_rate=" << cpu_w / total_w << "\n";
  std::cout << "z=" << local_w * 100 / num_node * T_local + remote_w * 100 / num_node * T_remote + cpu_w * 100 / num_node * T_cpu << "\n";

  block_placement = Tensor::CreateShm(Constant::kCollCachePlacementShmName, kU8, {static_cast<size_t>(num_block)}, "coll_cache_block_placement");

  for (int i = 0; i < num_device; i++) {
    block_placement->Ptr<uint8_t>()[i] = (1 << i);
  }
  block_placement->Ptr<uint8_t>()[num_device] = 0;
}

void PartRepSolver::Solve(std::vector<int> device_to_stream,
                            std::vector<int> device_to_cache_percent,
                            std::string mode, double T_local, double T_remote,
                            double T_cpu) {
  CHECK(RunConfig::coll_cache_link_desc._topo_type != AsymmLinkDesc::kHardWiredAsymm) << "PartRepSolver does not support asymm link topo";
  CHECK(std::accumulate(device_to_stream.begin(), device_to_stream.end(), 0, std::plus<>()) == 0);
  const int num_device = device_to_stream.size();
  const int num_block = num_device + 2;
  const IdType num_node = stream_freq_list->Shape()[1];
  // the freq list must already be sorted
  // now calculate the boundary
  const IdType num_cached_nodes = num_node * (device_to_cache_percent[0] / (double)100);

  LOG(ERROR) << "num_cached_nodes = " << num_cached_nodes;

  const IdType * freq_array = stream_freq_list->Ptr<IdType>();

  const IdType partition_size = (num_device == 1) ? num_cached_nodes : std::min(num_cached_nodes, (num_node - num_cached_nodes)/(num_device-1));
  const IdType replicate_size = num_cached_nodes - partition_size;
  CHECK_LE(replicate_size + partition_size * num_device, num_node);
  const IdType cpu_size = num_node - replicate_size - partition_size * num_device;

  double rep_w = 0, cpu_w = 0, total_w = 0;
  // block 0 -> replication
  // block 1-n -> partition,
  // block n+1 -> cpu
#pragma omp parallel for num_threads(RunConfig::omp_thread_num) reduction(+ : rep_w, cpu_w, total_w)
  for (IdType rank = 0; rank < num_node; rank++) {
    IdType node_id = stream_id_list->Ptr<IdType>()[rank];
    if (rank < replicate_size) {
      nid_to_block->Ptr<IdType>()[node_id] = 0;
      rep_w += freq_array[rank];
    } else if (rank < replicate_size + partition_size * num_device) {
      nid_to_block->Ptr<IdType>()[node_id] = (rank % num_device) + 1;
    } else {
      nid_to_block->Ptr<IdType>()[node_id] = num_device + 1;
      cpu_w += freq_array[rank];
    }
    total_w += freq_array[rank];
  }
  double partition_w = total_w - cpu_w - rep_w;
  double local_w = rep_w + partition_w / num_device;
  double remote_w = partition_w / num_device * (num_device - 1);
  std::cout << "coll_cache:optimal_rep_storage=" << replicate_size / (double)num_node << "\n";
  std::cout << "coll_cache:optimal_part_storage=" << partition_size * num_device / (double)num_node << "\n";
  std::cout << "coll_cache:optimal_cpu_storage=" << cpu_size / (double)num_node << "\n";
  std::cout << "coll_cache:optimal_local_storage=" << (replicate_size + partition_size) / (double)num_node << "\n";
  std::cout << "coll_cache:optimal_remote_storage=" << partition_size * (num_device - 1) / (double)num_node << "\n";
  std::cout << "coll_cache:optimal_local_rate=" << local_w / total_w << "\n";
  std::cout << "coll_cache:optimal_remote_rate=" << remote_w / total_w << "\n";
  std::cout << "coll_cache:optimal_cpu_rate=" << cpu_w / total_w << "\n";
  std::cout << "z=" << local_w * 100 / num_node * T_local + remote_w * 100 / num_node * T_remote + cpu_w * 100 / num_node * T_cpu << "\n";

  block_placement = Tensor::CreateShm(Constant::kCollCachePlacementShmName, kU8, {static_cast<size_t>(num_block)}, "coll_cache_block_placement");

  block_placement->Ptr<uint8_t>()[0] = (1 << num_device) - 1;
  for (int i = 0; i < num_device; i++) {
    block_placement->Ptr<uint8_t>()[i + 1] = (1 << i);
  }
  block_placement->Ptr<uint8_t>()[num_device + 1] = 0;
}

void RepSolver::Solve(std::vector<int> device_to_stream,
                      std::vector<int> device_to_cache_percent,
                      std::string mode, double T_local, double T_cpu) {
  CHECK(std::accumulate(device_to_stream.begin(), device_to_stream.end(), 0, std::plus<>()) == 0);
  const int num_device = device_to_stream.size();
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

  block_placement = Tensor::CreateShm(Constant::kCollCachePlacementShmName, kU8, {static_cast<size_t>(num_block)}, "coll_cache_block_placement");
  block_placement->Ptr<uint8_t>()[0] = (1 << num_device) - 1;
  block_placement->Ptr<uint8_t>()[1] = 0;
}

void CliquePartSolver::Solve(std::vector<int> device_to_stream,
                             std::vector<int> device_to_cache_percent,
                             std::string mode, double T_local, double T_cpu) {
  CHECK(stream_id_list->Shape()[0] == 1);
  CHECK(stream_freq_list->Shape()[0] == 1);
  CHECK(std::accumulate(device_to_stream.begin(), device_to_stream.end(), 0, std::plus<>()) == 0);
  const size_t num_device = device_to_stream.size();
  const int num_clique = RoundUpDiv<size_t>(num_device, clique_size);
  const int num_block = clique_size + 1;
  const IdType num_node = stream_freq_list->Shape()[1];
  // the freq list must already be sorted
  // now calculate the boundary
  const IdType num_cached_nodes = num_node * (device_to_cache_percent[0] / (double)100);

  // LOG(ERROR) << "num_cached_nodes = " << num_cached_nodes;
  CHECK_EQ(stream_freq_list->Type(), kI32);
  const IdType * freq_array = stream_freq_list->Ptr<IdType>();

  const IdType partition_size = std::min(num_cached_nodes, num_node / clique_size);

  double cpu_w = 0, total_w = 0;

#pragma omp parallel for num_threads(RunConfig::omp_thread_num) reduction(+ : cpu_w, total_w)
  for (IdType rank = 0; rank < num_node; rank++) {
    IdType node_id = stream_id_list->Ptr<IdType>()[rank];
    IdType block_id = clique_size;
    if (rank < partition_size * clique_size) {
      block_id = rank % clique_size;
    } else {
      block_id = clique_size;
      cpu_w += freq_array[rank];
    }
    total_w += freq_array[rank];
    nid_to_block->Ptr<IdType>()[node_id] = block_id;
  }
  double local_w = (total_w - cpu_w) / clique_size;
  double remote_w = (total_w - cpu_w) / clique_size * (clique_size - 1);
  double one_partition_w = local_w;

  std::vector<double> z_list(num_device, 0);

  for (IdType dev_id = 0; dev_id < num_device; dev_id++) {
    z_list[dev_id] += local_w * 100 / num_node * T_local;
    auto & link_list = RunConfig::coll_cache_link_desc.link_src[dev_id];
    auto & link_time_list = RunConfig::coll_cache_link_desc.link_time[dev_id];
    for (IdType link_id = 0; link_id < link_list.size(); link_id++) {
      if (link_list[link_id].size() == 0) continue;
      z_list[dev_id] += one_partition_w * 100 / num_node * link_time_list[link_id];
    }
    z_list[dev_id] += cpu_w * 100 / num_node * T_cpu;
  }

  std::cout << "coll_cache:optimal_rep_storage=" << 0 << "\n";
  std::cout << "coll_cache:optimal_part_storage=" << partition_size * clique_size / (double)num_node << "\n";
  std::cout << "coll_cache:optimal_cpu_storage=" << 1 - (partition_size * clique_size / (double)num_node) << "\n";
  std::cout << "coll_cache:optimal_local_storage=" << partition_size / (double)num_node << "\n";
  std::cout << "coll_cache:optimal_remote_storage=" << partition_size * (clique_size - 1) / (double)num_node << "\n";
  std::cout << "coll_cache:optimal_local_rate=" << local_w / total_w << "\n";
  std::cout << "coll_cache:optimal_remote_rate=" << remote_w / total_w << "\n";
  std::cout << "coll_cache:optimal_cpu_rate=" << cpu_w / total_w << "\n";
  std::cout << "z=";
  FOR_LOOP(dev_id, num_device) { std::cout << z_list[dev_id] << ","; }
  std::cout << "\n";

  auto dev_id_to_clique_id = [&](IdType dev_id) {
    return dev_id / clique_size;
  };
  auto block_id_to_storage_dev_in_clique = [&](IdType block_id, IdType clique_id) {
    return clique_id * clique_size + block_id;
  };
  auto block_id_to_storage_bitmap = [&](IdType block_id) {
    uint8_t storage = 0;
    for (IdType clique_id = 0; clique_id < num_clique; clique_id++) {
      storage |= 1 << block_id_to_storage_dev_in_clique(block_id, clique_id);
    }
    return storage;
  };
  auto block_id_to_src_dev = [&](IdType block_id, IdType dst_dev) {
    IdType clique_id = dev_id_to_clique_id(dst_dev);
    return block_id_to_storage_dev_in_clique(block_id, clique_id);
  };

  block_placement = Tensor::CreateShm(Constant::kCollCachePlacementShmName, kU8, {static_cast<size_t>(num_block)}, "coll_cache_block_placement");
  block_access_from = Tensor::CreateShm(Constant::kCollCacheAccessShmName, kU8, {num_device, static_cast<size_t>(num_block)}, "coll_cache_advise");
  TensorView<uint8_t> block_access_from_array(block_access_from);
  // there are clique_size + 1 blocks; first clique_size block is clique-partitioned, the last is on cpu
  for (int block_id = 0; block_id < clique_size; block_id++) {
    block_placement->Ptr<uint8_t>()[block_id] = block_id_to_storage_bitmap(block_id);
    for (IdType dev_id = 0; dev_id < num_device; dev_id++) {
      block_access_from_array[dev_id][block_id].ref() = block_id_to_src_dev(block_id, dev_id);
    }
  }
  // the one on cpu
  block_placement->Ptr<uint8_t>()[clique_size] = 0;
  for (IdType dev_id = 0; dev_id < num_device; dev_id++) {
    block_access_from_array[dev_id][clique_size].ref() = num_device;
  }

  FOR_LOOP(block_id, num_block) {
    // std::ios_base::fmtflags f( std::cerr.flags() );
    std::cerr << "block " << block_id
    //           << std::fixed << std::setw(8) << std::setprecision(6)
    //           << ", density=" << block_density_array[block_id]
    //           << std::fixed << std::setw(8) << std::setprecision(3)
    //           << ", freq=" << block_freq_array[block_id][0].ref()
    ;
    std::bitset<8> bs(block_placement->CPtr<uint8_t>()[block_id]);
    std::cerr << " storage is " << bs << "\n";
    // std::cerr.flags(f);
    std::cerr << "\taccess is\t";
    for (IdType dev_id = 0; dev_id < num_device; dev_id++) {
      std::cerr << int(block_access_from_array[dev_id][block_id].ref()) << "\t";
    }
    std::cerr << "\n";
  }
}

void CliquePartByDegreeSolver::Solve(std::vector<int> device_to_stream,
                             std::vector<int> device_to_cache_percent,
                             std::string mode, double T_local, double T_cpu) {
  CHECK(ranking_nodes->Shape().size() == 1);
  CHECK(stream_id_list->Shape()[0] == 1);
  CHECK(stream_freq_list->Shape()[0] == 1);
  CHECK(std::accumulate(device_to_stream.begin(), device_to_stream.end(), 0, std::plus<>()) == 0);
  const size_t num_device = device_to_stream.size();
  const int num_clique = RoundUpDiv<size_t>(num_device, clique_size);
  const int num_block = clique_size + 1;
  const IdType num_node = stream_freq_list->Shape()[1];
  // the freq list must already be sorted
  // now calculate the boundary
  const IdType num_cached_nodes = num_node * (device_to_cache_percent[0] / (double)100);

  // LOG(ERROR) << "num_cached_nodes = " << num_cached_nodes;
  CHECK_EQ(stream_freq_list->Type(), kI32);

  // map node id to freq
  TensorPtr node_id_to_freq = Tensor::Empty(kI32, {num_node}, CPU_CLIB(), "");
  {
    const IdType * freq_id_array = stream_id_list->Ptr<IdType>();
    const IdType * freq_array = stream_freq_list->Ptr<IdType>();
    #pragma omp parallel for num_threads(RunConfig::omp_thread_num)
    for (size_t i = 0; i < num_node; i++) {
      IdType node_id = freq_id_array[i];
      IdType freq = freq_array[i];
      node_id_to_freq->Ptr<IdType>()[node_id] = freq;
    }
  }

  // calculate partition
  const IdType partition_size = std::min(num_cached_nodes, num_node / clique_size);

  // distribute nodes onto GPU/CPU
  double cpu_w = 0, total_w = 0;

#pragma omp parallel for num_threads(RunConfig::omp_thread_num) reduction(+ : cpu_w, total_w)
  for (IdType rank = 0; rank < num_node; rank++) {
    IdType node_id = ranking_nodes->CPtr<IdType>()[rank];
    IdType block_id = clique_size;
    IdType freq = node_id_to_freq->CPtr<IdType>()[node_id];
    if (rank < partition_size * clique_size) {
      block_id = rank % clique_size;
    } else {
      block_id = clique_size;
      cpu_w += freq;
    }
    total_w += freq;
    nid_to_block->Ptr<IdType>()[node_id] = block_id;
  }
  double local_w = (total_w - cpu_w) / clique_size;
  double remote_w = (total_w - cpu_w) / clique_size * (clique_size - 1);
  double one_partition_w = local_w;

  std::vector<double> z_list(num_device, 0);

  for (IdType dev_id = 0; dev_id < num_device; dev_id++) {
    z_list[dev_id] += local_w * 100 / num_node * T_local;
    auto & link_list = RunConfig::coll_cache_link_desc.link_src[dev_id];
    auto & link_time_list = RunConfig::coll_cache_link_desc.link_time[dev_id];
    for (IdType link_id = 0; link_id < link_list.size(); link_id++) {
      if (link_list[link_id].size() == 0) continue;
      z_list[dev_id] += one_partition_w * 100 / num_node * link_time_list[link_id];
    }
    z_list[dev_id] += cpu_w * 100 / num_node * T_cpu;
  }

  std::cout << "coll_cache:optimal_rep_storage=" << 0 << "\n";
  std::cout << "coll_cache:optimal_part_storage=" << partition_size * clique_size / (double)num_node << "\n";
  std::cout << "coll_cache:optimal_cpu_storage=" << 1 - (partition_size * clique_size / (double)num_node) << "\n";
  std::cout << "coll_cache:optimal_local_storage=" << partition_size / (double)num_node << "\n";
  std::cout << "coll_cache:optimal_remote_storage=" << partition_size * (clique_size - 1) / (double)num_node << "\n";
  std::cout << "coll_cache:optimal_local_rate=" << local_w / total_w << "\n";
  std::cout << "coll_cache:optimal_remote_rate=" << remote_w / total_w << "\n";
  std::cout << "coll_cache:optimal_cpu_rate=" << cpu_w / total_w << "\n";
  std::cout << "z=";
  FOR_LOOP(dev_id, num_device) { std::cout << z_list[dev_id] << ","; }
  std::cout << "\n";

  auto dev_id_to_clique_id = [&](IdType dev_id) {
    return dev_id / clique_size;
  };
  auto block_id_to_storage_dev_in_clique = [&](IdType block_id, IdType clique_id) {
    return clique_id * clique_size + block_id;
  };
  auto block_id_to_storage_bitmap = [&](IdType block_id) {
    uint8_t storage = 0;
    for (IdType clique_id = 0; clique_id < num_clique; clique_id++) {
      storage |= 1 << block_id_to_storage_dev_in_clique(block_id, clique_id);
    }
    return storage;
  };
  auto block_id_to_src_dev = [&](IdType block_id, IdType dst_dev) {
    IdType clique_id = dev_id_to_clique_id(dst_dev);
    return block_id_to_storage_dev_in_clique(block_id, clique_id);
  };

  block_placement = Tensor::CreateShm(Constant::kCollCachePlacementShmName, kU8, {static_cast<size_t>(num_block)}, "coll_cache_block_placement");
  block_access_from = Tensor::CreateShm(Constant::kCollCacheAccessShmName, kU8, {num_device, static_cast<size_t>(num_block)}, "coll_cache_advise");
  TensorView<uint8_t> block_access_from_array(block_access_from);
  // there are clique_size + 1 blocks; first clique_size block is clique-partitioned, the last is on cpu
  for (int block_id = 0; block_id < clique_size; block_id++) {
    block_placement->Ptr<uint8_t>()[block_id] = block_id_to_storage_bitmap(block_id);
    for (IdType dev_id = 0; dev_id < num_device; dev_id++) {
      block_access_from_array[dev_id][block_id].ref() = block_id_to_src_dev(block_id, dev_id);
    }
  }
  // the one on cpu
  block_placement->Ptr<uint8_t>()[clique_size] = 0;
  for (IdType dev_id = 0; dev_id < num_device; dev_id++) {
    block_access_from_array[dev_id][clique_size].ref() = num_device;
  }

  FOR_LOOP(block_id, num_block) {
    // std::ios_base::fmtflags f( std::cerr.flags() );
    std::cerr << "block " << block_id
    //           << std::fixed << std::setw(8) << std::setprecision(6)
    //           << ", density=" << block_density_array[block_id]
    //           << std::fixed << std::setw(8) << std::setprecision(3)
    //           << ", freq=" << block_freq_array[block_id][0].ref()
    ;
    std::bitset<8> bs(block_placement->CPtr<uint8_t>()[block_id]);
    std::cerr << " storage is " << bs << "\n";
    // std::cerr.flags(f);
    std::cerr << "\taccess is\t";
    for (IdType dev_id = 0; dev_id < num_device; dev_id++) {
      std::cerr << int(block_access_from_array[dev_id][block_id].ref()) << "\t";
    }
    std::cerr << "\n";
  }
}

} // namespace coll_cache
}
}