#include "common/common.h"
#include "common/run_config.h"
#include "common/cpu/mmap_cpu_device.h"
#include "common/coll_cache/optimal_solver_class.h"
#include "solve_func_class.h"
#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>
#include <sys/fcntl.h>
#include <unistd.h>

using namespace samgraph;
using namespace samgraph::common;
using namespace samgraph::common::coll_cache;
#define FOR_LOOP(iter, len) for (uint32_t iter = 0; iter < (len); iter++)

namespace {
template<typename FROM_T, typename TO_T>
inline TO_T convert_datatype(FROM_T val) {
  return static_cast<TO_T>(val);
}

template<typename FROM_T, typename TO_T>
void convertTo(const FROM_T* src, TO_T* dst, size_t num_elem) {
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for (size_t i = 0; i < num_elem; i++) {
    dst[i] = src[i];
  }
}
template<typename FROM_T>
void convertTo(const FROM_T* src, TensorPtr dst_tensor) {
  switch (dst_tensor->Type()) {
    case kI32: convertTo(src, dst_tensor->Ptr<int32_t>(), dst_tensor->NumItem()); break;
    case kU8:  convertTo(src, dst_tensor->Ptr<uint8_t>(), dst_tensor->NumItem()); break;
    case kI64: convertTo(src, dst_tensor->Ptr<int64_t>(), dst_tensor->NumItem()); break;
    case kI8:  convertTo(src, dst_tensor->Ptr<int8_t>(),  dst_tensor->NumItem()); break;
    case kF32: convertTo(src, dst_tensor->Ptr<float>(),   dst_tensor->NumItem()); break;
    case kF64: convertTo(src, dst_tensor->Ptr<double>(),  dst_tensor->NumItem()); break;
    default:
      CHECK(false);
  }
}

TensorPtr convertTo(TensorPtr src, DataType dst_dtype) {
  auto ctx = src->Ctx();
  if (ctx.device_type == samgraph::common::kMMAP && ctx.device_id == MMAP_RO_DEVICE) {
    ctx.device_id = MMAP_RW_DEVICE;
  }
  auto dst_tensor = Tensor::Empty(dst_dtype, src->Shape(), ctx, src->Name());
  switch (src->Type()) {
    case kI32: convertTo(src->CPtr<int32_t>(), dst_tensor); break;
    case kU8:  convertTo(src->CPtr<uint8_t>(), dst_tensor); break;
    case kI64: convertTo(src->CPtr<int64_t>(), dst_tensor); break;
    case kI8:  convertTo(src->CPtr<int8_t>(),  dst_tensor); break;
    case kF32: convertTo(src->CPtr<float>(),   dst_tensor); break;
    case kF64: convertTo(src->CPtr<double>(),  dst_tensor); break;
    default:
      CHECK(false);
  }
  return dst_tensor;
}
}

#define METHOD_TYPE( F ) \
  F(coll_cache_single_stream) \
  F(coll_intuitive) \
  F(partition) \
  F(part_rep) \
  F(rep) \
  F(coll_cache) \
  F(coll_cache_asymm_link) \
  F(selfish) \
  F(part_rep_diff_freq) \
  F(clique_global_freq) \
  F(clique_local_freq) \
  F(local_int) \

#define F(name) k_##name,
enum MethodType {METHOD_TYPE( F ) kNumMethodType };
#undef F

#define F(name) #name ,
std::vector<std::string> method_names = { METHOD_TYPE( F ) };
#undef F

#define F(name) {#name , k_##name},
std::unordered_map<std::string, MethodType> method_name_mapping {
  METHOD_TYPE( F )
};
#undef F

template<typename T>
using vec = std::vector<T>;

int main(int argc, char** argv) {
  RunConfig::omp_thread_num = sysconf(_SC_NPROCESSORS_ONLN) / 2;
  RunConfig::num_train_worker = 1;
  RunConfig::coll_cache_hyperparam_T_local  = 1;
  RunConfig::coll_cache_hyperparam_T_remote = 438 / (double)213;  // performance on A100
  RunConfig::coll_cache_hyperparam_T_cpu    = 438 / (double)11.8; // performance on A100

  int cache_percent = -1;

  // RunConfig::coll_cache_hyperparam_T_remote = 330 / (double)42;  // performance on V100
  // RunConfig::coll_cache_hyperparam_T_cpu    = 330 / (double)11; // performance on V100

  std::vector<std::string> file_freq, file_id;
  std::vector<int> stream_mapping;
  std::string method = "coll_intuitive";
  std::string gpu = "A100";
  int clique_size = 1;
  CLI::App _app;
  _app.add_option("--fi,--file-id", file_id, "ranked id file")->required();
  _app.add_option("--ff,--file-freq", file_freq, "freq file")->required();
  _app.add_option("--sm,--stream-mapping", stream_mapping, "mapping of stream");
  _app.add_option("--nt,--num-train-worker", RunConfig::num_train_worker, "num trainer")->required();
  _app.add_option("-m,--method", method, "cache method")->check(CLI::IsMember(method_names));
  _app.add_option("-g,--gpu", gpu, "gpu model")->check(CLI::IsMember({"A100","V100"}));
  _app.add_option("-s,--slot", RunConfig::coll_cache_num_slot, "num slots");
  _app.add_option("-c,--cache-percent", cache_percent, "cache percent");
  _app.add_option("--coe", RunConfig::coll_cache_coefficient, "coll cache coefficient");
  _app.add_option("--clique", clique_size);
  try {
    _app.parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return _app.exit(e);
  }

  if (gpu == "A100") {
    RunConfig::coll_cache_hyperparam_T_remote = 438 / (double)213;  // performance on A100
    RunConfig::coll_cache_hyperparam_T_cpu    = 438 / (double)11.8; // performance on A100
  } else if (gpu == "V100") {
    RunConfig::coll_cache_hyperparam_T_remote = 330 / (double)42;  // performance on V100
    RunConfig::coll_cache_hyperparam_T_cpu    = 330 / (double)11; // performance on V100
  }
  // for 3 or 4 trainer, it must be hard wire settings: num link = trainer - 1; src per link = 1
  // for 6 or 8 trainer, depends on gpu type:
  //   A100, num link = 1; src per link = trainer -1
  //   V100, we should hand write the topo
  vec<vec<vec<int>>> link_src;
  vec<vec<double>> link_time;
  if (gpu == "V100" && RunConfig::num_train_worker == 8) {
  } else {
    int num_link = 1;
    int src_per_link = RunConfig::num_train_worker - 1;
    if (RunConfig::num_train_worker <= 4) {
      std::swap(num_link, src_per_link);
    }
    link_src = vec<vec<vec<int>>>(RunConfig::num_train_worker, vec<vec<int>>(num_link, vec<int>(src_per_link)));
    link_time = vec<vec<double>>(RunConfig::num_train_worker, vec<double>(num_link, RunConfig::coll_cache_hyperparam_T_remote));
    for (size_t dst_dev = 0; dst_dev < RunConfig::num_train_worker; dst_dev++) {
      for (size_t src_link = 0; src_link < num_link; src_link++) {
        std::cout << dst_dev << " : link #" << src_link << " : ";
        for (size_t src_dev = 0; src_dev < src_per_link; src_dev++) {
          link_src[dst_dev][src_link][src_dev] = (dst_dev + 1 + src_link + src_dev) % RunConfig::num_train_worker;
          std::cout << link_src[dst_dev][src_link][src_dev] << ",";
        }
        std::cout << "\n";
      }
    }
  }

  CHECK_EQ(file_freq.size(), file_id.size());
  size_t num_stream = file_freq.size();
  // if (num_stream == 8) {
  //   RunConfig::coll_cache_num_slot = 10;
  // }
  CHECK(num_stream > 0);
  if (stream_mapping.size() == 0) {
    if (num_stream == 1) {
      stream_mapping = std::vector<int>(RunConfig::num_train_worker, 0);
    } else if (RunConfig::num_train_worker % num_stream == 0) {
      stream_mapping = std::vector<int>(RunConfig::num_train_worker, 0);
      for (size_t d = 0; d < RunConfig::num_train_worker; d++) {
        stream_mapping[d] = d % num_stream;
      }
    } else {
      CHECK(false);
    }
  } else {
    CHECK_EQ(stream_mapping.size(), RunConfig::num_train_worker);
    CHECK_LE(*std::max_element(stream_mapping.begin(), stream_mapping.end()), num_stream);
  }
  std::cout << "Stream Mapping: ";
  for (size_t d = 0; d < RunConfig::num_train_worker; d++) {
    std::cout << stream_mapping[d] << " ";
  }
  std::cout << "\n";

  size_t num_nodes = 0;
  {
    size_t nbytes;
    int fd = cpu::MmapCPUDevice::OpenFile(file_id[0], &nbytes);
    close(fd);
    num_nodes = nbytes / sizeof(IdType);
  }

  auto id_tensor = Tensor::Empty(kI32, {num_stream, num_nodes}, MMAP(MMAP_RW_DEVICE), "");
  auto freq_tensor = Tensor::Empty(kF32, {num_stream, num_nodes}, MMAP(MMAP_RW_DEVICE), "");
  for (size_t stream_id = 0; stream_id < file_freq.size(); stream_id++) {
    {
      int fd = cpu::MmapCPUDevice::OpenFile(file_id[stream_id]);
      read(fd, id_tensor->Ptr<IdType>() + num_nodes * stream_id, sizeof(IdType) * num_nodes);
      close(fd);
    }{
      int fd = cpu::MmapCPUDevice::OpenFile(file_freq[stream_id]);
      read(fd, freq_tensor->Ptr<IdType>() + num_nodes * stream_id, sizeof(IdType) * num_nodes);
      close(fd);
    }
  }
  freq_tensor = convertTo(freq_tensor, kI32); // solver require float tensor for now.
  auto nid_to_block = Tensor::Empty(kI32, {num_nodes}, CPU(CPU_CLIB_MALLOC_DEVICE), "");
  TensorPtr block_placement;

  // if (method == "local_int") {
  //   coll_cache::solve_local_intuitive(id_tensor, freq_tensor, num_nodes, stream_mapping,
  //       {},
  //       nid_to_block, block_placement, "BIN",
  //       RunConfig::coll_cache_hyperparam_T_local,
  //       RunConfig::coll_cache_hyperparam_T_remote,
  //       RunConfig::coll_cache_hyperparam_T_cpu);
  //   return 0;
  // }
  int cache_begin = 0, cache_end = 100;
  if (cache_percent != -1) {
    cache_begin = cache_percent;
    cache_end = cache_percent;
  }
  CollCacheSolver * solver = nullptr;
  MethodType method_t = method_name_mapping[method];
  switch (method_t) {
    // single stream(global shuffle)
    case k_part_rep: solver = new PartRepSolver; break;
    case k_partition: solver = new PartitionSolver; break;
    case k_coll_intuitive: solver = new IntuitiveSolver; break;
    case k_rep : solver = new RepSolver; break;
    case k_coll_cache_single_stream:
    // multi stream(local shuffle)
    case k_coll_cache: solver = new OptimalSolver; break;
    case k_coll_cache_asymm_link: solver = new OptimalAsymmLinkSolver(link_src, link_time); break;
    case k_selfish : solver = new SelfishSolver; break;
    case k_part_rep_diff_freq: solver = new PartRepMultiStream; break;
    case k_clique_global_freq: solver = new CliqueGlobalFreqSolver(clique_size); break;
    case k_clique_local_freq: solver = new CliqueLocalFreqSolver(clique_size); break;
    default: CHECK(false);
  }
  solver->Build(id_tensor, freq_tensor, stream_mapping, num_nodes, nid_to_block);
  for (int cache_percent = cache_begin; cache_percent <= cache_end; cache_percent++) {
    RunConfig::cache_percentage = cache_percent / 100.0;
    std::cout << "cache_percent=" << cache_percent << "\n";
    auto dev_cache_size_list = std::vector<int> (RunConfig::num_train_worker, cache_percent);
    solver->Solve(stream_mapping, dev_cache_size_list, "BIN",
          RunConfig::coll_cache_hyperparam_T_local,
          RunConfig::coll_cache_hyperparam_T_remote,
          RunConfig::coll_cache_hyperparam_T_cpu);
  }
  return 0;
}