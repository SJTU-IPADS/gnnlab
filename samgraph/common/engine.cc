#include "engine.h"

#include <cstdlib>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <unordered_map>

#include "common.h"
#include "config.h"
#include "cpu/cpu_engine.h"
#include "cuda/cuda_engine.h"
#include "logging.h"
#include "timer.h"

namespace samgraph {
namespace common {

Engine* Engine::_engine = nullptr;

void Engine::Create(int device) {
  if (_engine) {
    return;
  }

  if (device < 0) {
    _engine = new cpu::CpuEngine;
  } else {
    _engine = new cuda::GpuEngine;
  }
}

void Engine::LoadGraphDataset() {
  Timer t;
  // Load graph dataset from disk by mmap and copy the graph
  // topology data into the target CUDA device.
  _dataset = new Dataset;
  std::unordered_map<std::string, size_t> meta;

  if (_dataset_path.back() != '/') {
    _dataset_path.push_back('/');
  }

  // Parse the meta data
  std::ifstream meta_file(_dataset_path + Config::kMetaFile);
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

  CHECK(meta.count(Config::kMetaNumNode) > 0);
  CHECK(meta.count(Config::kMetaNumEdge) > 0);
  CHECK(meta.count(Config::kMetaFeatDim) > 0);
  CHECK(meta.count(Config::kMetaNumClass) > 0);
  CHECK(meta.count(Config::kMetaNumTrainSet) > 0);
  CHECK(meta.count(Config::kMetaNumTestSet) > 0);
  CHECK(meta.count(Config::kMetaNumValidSet) > 0);

  _dataset->num_node = meta[Config::kMetaNumNode];
  _dataset->num_edge = meta[Config::kMetaNumEdge];
  _dataset->num_class = meta[Config::kMetaNumClass];

  _dataset->indptr = Tensor::FromMmap(
      _dataset_path + Config::kInptrFile, DataType::kI32,
      {meta[Config::kMetaNumNode] + 1}, _sample_device, "dataset.indptr");
  _dataset->indices = Tensor::FromMmap(
      _dataset_path + Config::kIndicesFile, DataType::kI32,
      {meta[Config::kMetaNumEdge]}, _sample_device, "dataset.indices");
  _dataset->feat =
      Tensor::FromMmap(_dataset_path + Config::kFeatFile, DataType::kF32,
                       {meta[Config::kMetaNumNode], meta[Config::kMetaFeatDim]},
                       CPU_DEVICE_MMAP_ID, "dataset.feat");
  _dataset->label = Tensor::FromMmap(
      _dataset_path + Config::kLabelFile, DataType::kI64,
      {meta[Config::kMetaNumNode]}, CPU_DEVICE_MMAP_ID, "dataset.label");
  _dataset->train_set = Tensor::FromMmap(
      _dataset_path + Config::kTrainSetFile, DataType::kI32,
      {meta[Config::kMetaNumTrainSet]}, CPU_DEVICE_ID, "dataset.train_set");
  _dataset->test_set = Tensor::FromMmap(
      _dataset_path + Config::kTestSetFile, DataType::kI32,
      {meta[Config::kMetaNumTestSet]}, CPU_DEVICE_ID, "dataset.test_set");
  _dataset->valid_set = Tensor::FromMmap(
      _dataset_path + Config::kValidSetFile, DataType::kI32,
      {meta[Config::kMetaNumValidSet]}, CPU_DEVICE_ID, "dataset.valid_set");

  double loading_time = t.Passed();
  LOG(INFO) << "SamGraph loaded dataset(" << _dataset_path << ") successfully ("
            << loading_time << " secs)";
}

bool Engine::IsAllThreadFinish(int total_thread_num) {
  int k = _joined_thread_cnt.fetch_add(0);
  return (k == total_thread_num);
};

}  // namespace common
}  // namespace samgraph
