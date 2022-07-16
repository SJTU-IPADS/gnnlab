#include "common/common.h"
#include "common/run_config.h"
#include "common/cpu/mmap_cpu_device.h"
#include "common/coll_cache/optimal_solver.h"
#include "solve_func.h"
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

int main(int argc, char** argv) {
  RunConfig::omp_thread_num = sysconf(_SC_NPROCESSORS_ONLN) / 2;
  RunConfig::num_train_worker = 1;
  RunConfig::coll_cache_hyperparam_T_local  = 1;
  RunConfig::coll_cache_hyperparam_T_remote = 438 / (double)213;  // performance on A100
  RunConfig::coll_cache_hyperparam_T_cpu    = 438 / (double)11.8; // performance on A100

  // RunConfig::coll_cache_hyperparam_T_remote = 330 / (double)42;  // performance on V100
  // RunConfig::coll_cache_hyperparam_T_cpu    = 330 / (double)11; // performance on V100

  std::vector<std::string> file_freq, file_id;
  std::vector<int> stream_mapping;
  std::string method = "coll_intuitive";
  std::string gpu = "A100";
  CLI::App _app;
  _app.add_option("--fi,--file-id", file_id, "ranked id file")->required();
  _app.add_option("--ff,--file-freq", file_freq, "freq file")->required();
  _app.add_option("--sm,--stream-mapping", stream_mapping, "mapping of stream");
  _app.add_option("--nt,--num-train-worker", RunConfig::num_train_worker, "num trainer")->required();
  _app.add_option("-m,--method", method, "cache method")->check(CLI::IsMember({
      "coll_cache_single_stream", 
      "coll_intuitive",
      "partition",
      "part_rep",
      "rep",
      "coll_cache",
      "selfish",
      "part_rep_diff_freq",
    }));
  _app.add_option("-g,--gpu", gpu, "gpu model")->check(CLI::IsMember({"A100","V100"}));
  _app.add_option("-s,--slot", RunConfig::coll_cache_num_slot, "num slots");
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

  for (int cache_percent = 0; cache_percent <= 100; cache_percent++) {
    RunConfig::cache_percentage = cache_percent / 100.0;
    std::cout << "cache_percent=" << cache_percent << "\n";
    if (method == "partition") {
      coll_cache::solve_partition(id_tensor, freq_tensor, num_nodes, stream_mapping,
          std::vector<int> (RunConfig::num_train_worker, cache_percent),
          nid_to_block, block_placement, "", 
          RunConfig::coll_cache_hyperparam_T_local,
          RunConfig::coll_cache_hyperparam_T_remote,
          RunConfig::coll_cache_hyperparam_T_cpu);
    } else if (method == "coll_cache_single_stream") {
      coll_cache::solve(id_tensor, freq_tensor, num_nodes, stream_mapping,
          std::vector<int> (RunConfig::num_train_worker, cache_percent),
          nid_to_block, block_placement, "BIN", 
          RunConfig::coll_cache_hyperparam_T_local,
          RunConfig::coll_cache_hyperparam_T_remote,
          RunConfig::coll_cache_hyperparam_T_cpu);
    } else if (method == "coll_intuitive") {
      coll_cache::solve_intuitive(id_tensor, freq_tensor, num_nodes, stream_mapping,
          std::vector<int> (RunConfig::num_train_worker, cache_percent),
          nid_to_block, block_placement, "", 
          RunConfig::coll_cache_hyperparam_T_local,
          RunConfig::coll_cache_hyperparam_T_remote,
          RunConfig::coll_cache_hyperparam_T_cpu);
    } else if (method == "part_rep") {
      coll_cache::solve_partition_rep(id_tensor, freq_tensor, num_nodes, stream_mapping,
          std::vector<int> (RunConfig::num_train_worker, cache_percent),
          nid_to_block, block_placement, "", 
          RunConfig::coll_cache_hyperparam_T_local,
          RunConfig::coll_cache_hyperparam_T_remote,
          RunConfig::coll_cache_hyperparam_T_cpu);
    } else if (method == "rep") {
      coll_cache::solve_rep(id_tensor, freq_tensor, num_nodes, stream_mapping,
          std::vector<int> (RunConfig::num_train_worker, cache_percent),
          nid_to_block, block_placement, "", 
          RunConfig::coll_cache_hyperparam_T_local,
          RunConfig::coll_cache_hyperparam_T_remote,
          RunConfig::coll_cache_hyperparam_T_cpu);
    } else if (method == "coll_cache") {
      coll_cache::solve(id_tensor, freq_tensor, num_nodes, stream_mapping,
          std::vector<int> (RunConfig::num_train_worker, cache_percent),
          nid_to_block, block_placement, "BIN", 
          RunConfig::coll_cache_hyperparam_T_local,
          RunConfig::coll_cache_hyperparam_T_remote,
          RunConfig::coll_cache_hyperparam_T_cpu);
    } else if (method == "selfish") {
      coll_cache::solve_selfish(id_tensor, freq_tensor, num_nodes, stream_mapping,
          std::vector<int> (RunConfig::num_train_worker, cache_percent),
          nid_to_block, block_placement, "BIN", 
          RunConfig::coll_cache_hyperparam_T_local,
          RunConfig::coll_cache_hyperparam_T_remote,
          RunConfig::coll_cache_hyperparam_T_cpu);
    } else if (method == "part_rep_diff_freq") {
      coll_cache::solve_partition_rep_diff_freq(id_tensor, freq_tensor, num_nodes, stream_mapping,
          std::vector<int> (RunConfig::num_train_worker, cache_percent),
          nid_to_block, block_placement, "BIN", 
          RunConfig::coll_cache_hyperparam_T_local,
          RunConfig::coll_cache_hyperparam_T_remote,
          RunConfig::coll_cache_hyperparam_T_cpu);
    }
  }
  return 0;
}