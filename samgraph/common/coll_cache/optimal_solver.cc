#include "../common.h"
#include "../run_config.h"
#include "../logging.h"
#include "../device.h"
#include "../cpu/mmap_cpu_device.h"
#include "ndarray.h"
#include <sys/mman.h>
#include <unistd.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <sys/fcntl.h>
#include <cstring>
#include <bitset>
#include "gurobi_c++.h"

namespace samgraph {
namespace common {
namespace coll_cache {

namespace block_builder {

static double alpha = 0;
const size_t & num_slot = RunConfig::coll_cache_num_slot;
const double & coefficient = RunConfig::coll_cache_coefficient;

int freq_to_slot_1(float freq, uint32_t rank, IdType num_node) {
  if (freq == 0) return num_slot - 1;
  if (freq >= alpha) return 0;
  double exp = std::log2(alpha / (double)freq) / std::log2(coefficient);
  int slot = (int)std::ceil(exp);
  slot = std::min(slot, (int)num_slot - 2);
  return slot;
}
int freq_to_slot_2(float freq, uint32_t rank, IdType num_node) {
  return rank * (uint64_t)num_slot / num_node;
}
std::function<int(float, uint32_t, IdType)> freq_to_slot = freq_to_slot_1;
// std::function<int(float, uint32_t)> freq_to_slot = freq_to_slot_2;

/**
 * each <slot freq> is mapped to a block, but the block may get too large.
 * so we split block into smaller blocks:
 * initially, each <slot freq> maps to a default block;
 * by inserting vertex into block, if the block exceed size limit, alloc a new
 * block (by atomic adding next_free_block) and update the <slot freq> mapping.
 */
static std::atomic_uint32_t next_free_block(0);
auto alloc_block = []() {
  return next_free_block.fetch_add(1);
};
static uint32_t max_size_per_block = 10000;

struct block_identifer {
  uint32_t current_block_id = -1;
  uint32_t current_num = 0;
  bool ignore_limit = false;
  std::atomic_flag latch;
  block_identifer() : latch() {}
  void set_ignore_limit() {
    while (latch.test_and_set()) {}
    ignore_limit = true;
    latch.clear();
  }
  uint32_t add_node() {
    while (latch.test_and_set()) {}
    uint32_t selected_block = -1;
    if (current_num < max_size_per_block || ignore_limit) {
      selected_block = current_block_id;
      current_num ++;
    } else {
      current_block_id = alloc_block();
      selected_block = current_block_id;
      current_num = 1;
    }
    latch.clear();
    return selected_block;
  }
};

void split_blocks(TensorPtr stream_id_list, TensorPtr stream_freq_list, const IdType num_node, 
    TensorPtr & block_density_tensor, TensorPtr & block_freq_tensor, const TensorPtr nid_to_block_tensor) {
  CHECK_EQ(stream_id_list->Shape().size(), 2);
  CHECK_EQ(stream_freq_list->Shape().size(), 2);
  CHECK_EQ(stream_id_list->Shape(), stream_freq_list->Shape());
  CHECK_EQ(stream_id_list->Shape()[1], num_node);
  IdType num_stream = stream_id_list->Shape()[0];
  auto cpu_ctx = CPU(CPU_CLIB_MALLOC_DEVICE);

  TensorPtr nid_to_rank_tensor  = Tensor::Empty(kI32, {num_node, num_stream}, cpu_ctx, "coll_cache.nid_to_rank");
  TensorPtr nid_to_slot_tensor  = Tensor::Empty(kI32, {num_node, num_stream}, cpu_ctx, "coll_cache.nid_to_slot");
  // nid_to_block_tensor = Tensor::Empty(kI32, {num_node}, cpu_ctx, "coll_cache.nid_to_block");
  CHECK(nid_to_block_tensor->Shape() == std::vector<size_t>{num_node});

  TensorView<uint32_t> nid_to_rank(nid_to_rank_tensor);
  TensorView<uint32_t> nid_to_slot(nid_to_slot_tensor);
  TensorView<uint32_t> nid_to_block(nid_to_block_tensor);

  uint32_t num_of_full_block = 1;
  for (int i = 0; i < num_stream; i++) { num_of_full_block *= num_slot; }
  // Tensor does not support non-native datatype, so use raw array
  block_identifer * slot_array_to_full_block_storage = (block_identifer*)
      Device::Get(cpu_ctx)->AllocWorkspace(cpu_ctx, num_of_full_block * sizeof(block_identifer));
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for (uint32_t slot_array_seq_id = 0; slot_array_seq_id < num_of_full_block; slot_array_seq_id++) {
    new (&slot_array_to_full_block_storage[slot_array_seq_id]) block_identifer;
    slot_array_to_full_block_storage[slot_array_seq_id].current_block_id = slot_array_seq_id;
  }
  TensorView<block_identifer> slot_array_to_full_block(slot_array_to_full_block_storage, std::vector<size_t>(num_stream, num_slot));
  next_free_block.store(num_of_full_block);
  // the last default block corresponds to zero freq, no need to split it.
  // slot_array_to_full_block_storage[num_of_full_block - 1].set_ignore_limit();

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
  LOG(INFO) << "mapping nid to rank...\n";
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for (uint32_t orig_rank = 0; orig_rank < num_node; orig_rank++) {
    for (uint32_t stream_idx = 0; stream_idx < num_stream; stream_idx++) {
      uint32_t nid = stream_id_list_view[stream_idx][orig_rank].ref();
      nid_to_rank[nid][stream_idx].ref() = orig_rank;
    }
  }
  // auto slot_list_to_block_id = [&block_density_array](const TensorView<uint32_t>& slot_array){
  //   return block_density_array[slot_array]._data - block_density_array._data;
  // };
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
    nid_to_block[nid].ref() = slot_array_to_full_block[nid_to_slot[nid]].ref().add_node();
  }
  Device::Get(cpu_ctx)->FreeWorkspace(cpu_ctx, slot_array_to_full_block_storage);

  /**
   * Sum frequency & density of each block
   */
  LOG(INFO) << "counting freq and density...\n";
  uint32_t total_num_blocks = next_free_block.load();
  block_density_tensor = Tensor::Empty(kF64, {total_num_blocks}, cpu_ctx, "coll_cache.block_density_tensor");
  block_freq_tensor    = Tensor::Empty(kF64, {total_num_blocks, num_stream}, cpu_ctx, "coll_cache.block_freq_tensor");

  std::memset(block_density_tensor->MutableData(), 0, block_density_tensor->NumBytes());
  std::memset(block_freq_tensor->MutableData(), 0, block_freq_tensor->NumBytes());

  TensorView<double> block_density_array(block_density_tensor);
  TensorView<double> block_freq_array(block_freq_tensor);
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for (uint32_t thread_idx = 0; thread_idx < RunConfig::omp_thread_num; thread_idx++) {
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
  LOG(INFO) << "averaging freq and density...\n";
  LOG(INFO) << block_density_tensor->NumItem();
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
}


}
namespace solver {

#define FOR_LOOP(iter, len) for (uint32_t iter = 0; iter < (len); iter++)
#define FOR_LOOP_1(iter, len) for (uint32_t iter = 1; iter < (len); iter++)
static_assert(sizeof(GRBVar) == sizeof(Id64Type), "size of GRBVar is not 8byte, cannot use tensor to hold it..");

class Solver {
 public:
  Solver(TensorPtr block_density_tensor, TensorPtr block_freq_tensor, 
         std::vector<int> device_to_stream, std::vector<int> device_to_cache_percent,
         std::string mode,
         double T_local = 2, double T_remote = 15, double T_cpu = 60) {
    CHECK(mode == "BIN");
    CHECK(block_density_tensor->Defined());
    this->block_density_tensor = block_density_tensor;
    this->block_freq_tensor = block_freq_tensor;
    this->block_density_array = (double*)block_density_tensor->MutableData();
    this->block_freq_array.rebuild(block_freq_tensor);
    this->device_to_stream = device_to_stream;
    this->cache_percent = device_to_cache_percent.at(0);
    this->T_local = T_local;
    this->T_remote = T_remote;
    this->T_cpu = T_cpu;
    this->mode = mode;
    num_device = device_to_stream.size();
    num_stream = block_freq_tensor->Shape().at(1);
    num_block = block_density_tensor->Shape().at(0);
    LOG(INFO) << "constructing optimal solver, device=" << num_device << ", stream=" << num_stream;
  }
 private:
  GRBVar z;
  TensorView<GRBVar> c_list;
  TensorView<GRBVar> x_list;
  std::vector<GRBLinExpr> time_list;
  std::vector<GRBLinExpr> local_weight_list;
  std::vector<double> total_weight_list;
  std::vector<GRBLinExpr> cpu_weight_list;

  const int & solver_num_thread = RunConfig::omp_thread_num;

  double * block_density_array;
  TensorView<double> block_freq_array;
  TensorPtr block_density_tensor, block_freq_tensor;
  std::vector<int> device_to_stream;
  uint32_t cache_percent = 14;
  std::string mode = "CONT";
  double T_local = 2, T_remote = 15, T_cpu = 60;

  uint32_t num_device;
  uint32_t num_stream;
  uint32_t num_block;

  inline bool ignore_block(uint32_t block_id, double weight) {
    return (weight == 0) && (block_id > 0);
  }

  void constraint_connect_c_x(GRBModel & model, uint32_t block_id) {
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
  }
  void constraint_connect_r_x(GRBModel & model, uint32_t block_id, uint32_t device_id) {
    if (mode == "CONT") {
      if (block_density_array[block_id] == 0) return;
      model.addConstr(c_list[block_id].ref() + x_list[block_id][device_id].ref() <= 1);
    }
  }
  void constraint_capacity(GRBModel & model, uint32_t device_id) {
    GRBLinExpr expr;
    FOR_LOOP(block_id, num_block) {
      if (block_density_array[block_id] == 0) continue;
      expr += x_list[block_id][device_id].ref() * block_density_array[block_id];
    }
    model.addConstr(expr <= cache_percent);
  }
  void constraint_time(GRBModel & model, uint32_t device_id) {
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
  }
 public:
  void solve_optimal(const TensorPtr block_placement) {
    std::cerr << num_block << " blocks, " << num_device << " devices\n";

    GRBEnv env = GRBEnv(true);
    env.set("LogFile", "cppsolver.log");
    // BarConvTol
    env.set(GRB_IntParam_Presolve, 0);
    env.set(GRB_IntParam_Method, 2);
    env.set(GRB_IntParam_Threads, solver_num_thread);
    env.set(GRB_DoubleParam_BarConvTol, 1e-2);
    env.set(GRB_DoubleParam_OptimalityTol, 1e-2);
    env.set(GRB_DoubleParam_MIPGap, 1e-3);
    env.start();

    GRBModel model = GRBModel(env);

    z = model.addVar(0.0, std::numeric_limits<double>::max(), 0.0, GRB_CONTINUOUS, "z");

    TensorPtr c_list_tensor = Tensor::Empty(kI64, {num_block}, CPU(CPU_CLIB_MALLOC_DEVICE), "c_list");
    TensorPtr x_list_tensor = Tensor::Empty(kI64, {num_block, num_device}, CPU(CPU_CLIB_MALLOC_DEVICE), "x_list");

    c_list.rebuild(c_list_tensor);
    x_list.rebuild(x_list_tensor);
    time_list.resize(num_device);
    local_weight_list.resize(num_device);
    total_weight_list.resize(num_device);
    cpu_weight_list.resize(num_device);

    std::cerr << "Add Var...\n";
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

    std::cerr << "Capacity...\n";
    FOR_LOOP(device_id, num_device) {constraint_capacity(model, device_id);}

    std::cerr << "Connect CPU...\n";
    FOR_LOOP(block_id, num_block) {constraint_connect_c_x(model, block_id);}

    std::cerr << "Connect Remote...\n";
    FOR_LOOP(block_id, num_block) {
      FOR_LOOP(device_id, num_device) {constraint_connect_r_x(model, block_id, device_id);}
    }

    std::cerr << "Time...\n";
    FOR_LOOP(device_id, num_device) {constraint_time(model, device_id);}

    model.setObjective(z + 0, GRB_MINIMIZE);

    model.optimize();

    // std::ofstream fs("solver_gurobi.out");
    // fs << z.get(GRB_DoubleAttr::GRB_DoubleAttr_X);
    // double max_gpu_time = 0;
    // FOR_LOOP(device_id, num_device) {
    //   if (time_list[device_id].getValue() > max_gpu_time) {
    //     max_gpu_time = time_list[device_id].getValue();
    //   }
    // }
    // fs << " " << max_gpu_time;
    // FOR_LOOP(device_id, num_device) {
    //   fs << " " << time_list[device_id].getValue();
    // }
    // fs.close();
    CHECK(num_device <= 8);
    // block_placement = Tensor::Empty(kU8, block_freq_tensor->Shape(), CPU(CPU_CLIB_MALLOC_DEVICE), "optimal_placement_block");
    CHECK(block_placement->Shape() == std::vector<size_t>{num_block});
    TensorView<uint8_t> block_placement_array(block_placement);
    LOG(INFO) << "Coll Cache init block placement array";
// #pragma omp parallel for num_threads(RunConfig::omp_thread_num)
    FOR_LOOP(block_id, num_block) {
      block_placement_array[block_id].ref() = 0;
      std::ios_base::fmtflags f( std::cerr.flags() );
      std::cerr << "block " << block_id
                << std::fixed << std::setw(8) << std::setprecision(6)
                << ", density=" << block_density_array[block_id]
                << std::fixed << std::setw(8) << std::setprecision(3)
                << ", freq=" << block_freq_array[block_id][0].ref();
      if (!ignore_block(block_id, block_density_array[block_id])) {
        FOR_LOOP(device_id, num_device) {
          uint8_t x_result = (uint8_t)std::round(x_list[block_id][device_id].ref().get(GRB_DoubleAttr::GRB_DoubleAttr_X));
          block_placement_array[block_id].ref() |= (x_result << device_id);
        }
      }
      std::bitset<8> bs(block_placement_array[block_id].ref());
      std::cerr << "  storage is " << bs << "\n";
      std::cerr.flags(f);
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
};

}

void solve(
    TensorPtr stream_id_list, TensorPtr stream_freq_list, const IdType num_node,
    std::vector<int> device_to_stream, std::vector<int> device_to_cache_percent,
    TensorPtr nid_to_block, TensorPtr & block_placement,
    std::string mode, double T_local, double T_remote, double T_cpu) {
  TensorPtr block_density_tensor;
  TensorPtr block_freq_tensor;
  block_builder::split_blocks(stream_id_list, stream_freq_list, num_node, block_density_tensor, block_freq_tensor, nid_to_block);

  block_placement = Tensor::CreateShm("coll_cache_block_placement", kU8, block_density_tensor->Shape(), "coll_cache_block_placement");

  solver::Solver s(block_density_tensor, block_freq_tensor, device_to_stream, device_to_cache_percent, mode, T_local, T_remote, T_cpu);
  s.solve_optimal(block_placement);
}

void solve_intuitive(TensorPtr stream_id_list, TensorPtr stream_freq_list, const IdType num_node,
    std::vector<int> device_to_stream, std::vector<int> device_to_cache_percent,
    TensorPtr nid_to_block, TensorPtr & block_placement,
    std::string mode, double T_local = 2, double T_remote = 15, double T_cpu = 60) {
  CHECK(stream_id_list->Shape()[0] == 1);
  CHECK(stream_freq_list->Shape()[0] == 1);
  CHECK(std::accumulate(device_to_stream.begin(), device_to_stream.end(), 0, std::plus<>()) == 0);
  const int num_device = device_to_stream.size();
  const int num_block = num_device + 2;
  // the freq list must already be sorted
  // now calculate the boundary
  const IdType num_cached_nodes = num_node * (device_to_cache_percent[0] / (double)100);
  IdType partition_size_min = 0;
  IdType partition_size_max = std::min(num_cached_nodes, (num_node - num_cached_nodes) / (num_device - 1));

  LOG(ERROR) << "num_cached_nodes = " << num_cached_nodes;
  LOG(ERROR) << "[" << partition_size_min << "," << partition_size_max << "]";

  // IdType partition_size = 0;

  auto partition_lb = [&](IdType partition_size) {
    return num_cached_nodes - partition_size;
  };
  auto partition_rb = [&](IdType partition_size) {
    return num_cached_nodes + partition_size * (num_device - 1) - 1;
  };

  // const double T_partition = (T_local + (num_device - 1) * T_remote) / num_device;
  const double mu = 1 + (T_cpu - T_remote) / (T_remote - T_local) * num_device;
  const IdType * freq_array = stream_freq_list->Ptr<IdType>();

  LOG(ERROR) << "mu = " << mu;

//   {
//     // analyze result of different partition size
//     for (IdType partition_size = partition_size_min; partition_size < partition_size_max; partition_size += (partition_size_max - partition_size_min)/100) {
//       double rep_w = 0, cpu_w = 0, total_w = 0;
//       IdType rep_d = 0, cpu_d = 0;
// #pragma omp parallel for num_threads(RunConfig::omp_thread_num) reduction(+ : rep_w, cpu_w, total_w, rep_d, cpu_d)
//       for (IdType rank = 0; rank < num_node; rank++) {
//         IdType node_id = stream_id_list->Ptr<IdType>()[rank];
//         if (rank < partition_lb(partition_size)) {
//           // replicate this
//           rep_w += freq_array[rank];
//           rep_d ++;
//         } else if (rank <= partition_rb(partition_size)) {
//         } else {
//           cpu_w += freq_array[rank];
//           cpu_d++;
//         }
//         total_w += freq_array[rank];
//       }
//       double local_w = rep_w + (total_w - cpu_w - rep_w) / num_device;
//       double remote_w = (total_w - cpu_w - rep_w) / num_device * (num_device - 1);
//       std::ios cout_state(nullptr);
//       cout_state.copyfmt(std::cout);
//       std::cout << std::fixed << std::setw(4) << std::setprecision(1);
//       // std::cout << "[" << partition_size << "](" << partition_size_min << ")";
//       std::cout << "[R " << rep_d * 100.0 / num_node << ",P " << (num_node - rep_d - cpu_d)*100.0/num_node << ",C" << cpu_d *100.0/num_node << "]";
//       std::cout << "[lb.freq=" << freq_array[partition_lb(partition_size)] << ",rb.freq=" << freq_array[partition_rb(partition_size)] << "]";
//       std::cout.copyfmt(cout_state);
//       std::cout << "local_rate=" << local_w / total_w << "\t";
//       std::cout << "remote_rate=" << remote_w / total_w << "\t";
//       std::cout << "cpu_rate=" << cpu_w / total_w << "\t";
//       std::cout << "z=" << local_w * 100 / num_node * T_local + remote_w * 100 / num_node * T_remote + cpu_w * 100 / num_node * T_cpu << "\n";
//     }
//   }

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
      LOG(ERROR) << "[" << partition_size_min << "," << partition_size_max << "]";
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
  std::cout << "coll_cache:optimal_local_rate=" << local_w / total_w << "\n";
  std::cout << "coll_cache:optimal_remote_rate=" << remote_w / total_w << "\n";
  std::cout << "coll_cache:optimal_cpu_rate=" << cpu_w / total_w << "\n";
  std::cout << "z=" << local_w * 100 / num_node * T_local + remote_w * 100 / num_node * T_remote + cpu_w * 100 / num_node * T_cpu << "\n";

  block_placement = Tensor::CreateShm("coll_cache_block_placement", kU8, {static_cast<size_t>(num_block)}, "coll_cache_block_placement");

  block_placement->Ptr<uint8_t>()[0] = (1 << num_device) - 1;
  for (int i = 0; i < num_device; i++) {
    block_placement->Ptr<uint8_t>()[i + 1] = (1 << i);
  }
  block_placement->Ptr<uint8_t>()[num_device + 1] = 0;
}

void solve_partition(TensorPtr stream_id_list, TensorPtr stream_freq_list, const IdType num_node,
    std::vector<int> device_to_stream, std::vector<int> device_to_cache_percent,
    TensorPtr nid_to_block, TensorPtr & block_placement,
    std::string _, double T_local = 2, double T_remote = 15, double T_cpu = 60) {
  CHECK(stream_id_list->Shape()[0] == 1);
  CHECK(stream_freq_list->Shape()[0] == 1);
  CHECK(std::accumulate(device_to_stream.begin(), device_to_stream.end(), 0, std::plus<>()) == 0);
  const int num_device = device_to_stream.size();
  const int num_block = num_device + 1;
  // the freq list must already be sorted
  // now calculate the boundary
  const IdType num_cached_nodes = num_node * (device_to_cache_percent[0] / (double)100);

  LOG(ERROR) << "num_cached_nodes = " << num_cached_nodes;

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
  std::cout << "coll_cache:optimal_local_rate=" << local_w / total_w << "\n";
  std::cout << "coll_cache:optimal_remote_rate=" << remote_w / total_w << "\n";
  std::cout << "coll_cache:optimal_cpu_rate=" << cpu_w / total_w << "\n";
  std::cout << "z=" << local_w * 100 / num_node * T_local + remote_w * 100 / num_node * T_remote + cpu_w * 100 / num_node * T_cpu << "\n";

  block_placement = Tensor::CreateShm("coll_cache_block_placement", kU8, {static_cast<size_t>(num_block)}, "coll_cache_block_placement");

  for (int i = 0; i < num_device; i++) {
    block_placement->Ptr<uint8_t>()[i] = (1 << i);
  }
  block_placement->Ptr<uint8_t>()[num_device] = 0;
}

}
}
}