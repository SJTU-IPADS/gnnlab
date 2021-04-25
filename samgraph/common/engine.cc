#include "engine.h"

#include <cstdlib>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <unordered_map>

#include "common.h"
#include "constant.h"
#include "cpu/cpu_engine.h"
#include "cuda/cuda_engine.h"
#include "logging.h"
#include "profiler.h"
#include "run_config.h"
#include "timer.h"

namespace samgraph {
namespace common {

Engine* Engine::_engine = nullptr;

void Engine::Report(uint64_t epoch, uint64_t step) {
  uint64_t key = Engine::GetBatchKey(epoch, step);
  Profiler::Get().Report(key);
}

void Engine::Create() {
  if (_engine) {
    return;
  }

  switch (RunConfig::sampler_ctx.device_type) {
    case kCPU:
      LOG(INFO) << "Use CPU Engine";
      _engine = new cpu::CPUEngine;
      break;
    case kGPU:
      LOG(INFO) << "Use GPU Engine";
      _engine = new cuda::GPUEngine;
      break;
    default:
      CHECK(0);
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
  std::ifstream meta_file(_dataset_path + Constant::kMetaFile);
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

  CHECK(meta.count(Constant::kMetaNumNode) > 0);
  CHECK(meta.count(Constant::kMetaNumEdge) > 0);
  CHECK(meta.count(Constant::kMetaFeatDim) > 0);
  CHECK(meta.count(Constant::kMetaNumClass) > 0);
  CHECK(meta.count(Constant::kMetaNumTrainSet) > 0);
  CHECK(meta.count(Constant::kMetaNumTestSet) > 0);
  CHECK(meta.count(Constant::kMetaNumValidSet) > 0);

  _dataset->num_node = meta[Constant::kMetaNumNode];
  _dataset->num_edge = meta[Constant::kMetaNumEdge];
  _dataset->num_class = meta[Constant::kMetaNumClass];

  _dataset->indptr =
      Tensor::FromMmap(_dataset_path + Constant::kInptrFile, DataType::kI32,
                       {meta[Constant::kMetaNumNode] + 1},
                       _sampler_ctx.device_type == kCPU ? MMAP() : _sampler_ctx,
                       "dataset.indptr");
  _dataset->indices =
      Tensor::FromMmap(_dataset_path + Constant::kIndicesFile, DataType::kI32,
                       {meta[Constant::kMetaNumEdge]},
                       _sampler_ctx.device_type == kCPU ? MMAP() : _sampler_ctx,
                       "dataset.indices");
  _dataset->feat = Tensor::FromMmap(
      _dataset_path + Constant::kFeatFile, DataType::kF32,
      {meta[Constant::kMetaNumNode], meta[Constant::kMetaFeatDim]}, MMAP(),
      "dataset.feat");
  _dataset->label =
      Tensor::FromMmap(_dataset_path + Constant::kLabelFile, DataType::kI64,
                       {meta[Constant::kMetaNumNode]}, MMAP(), "dataset.label");

  _dataset->train_set = Tensor::FromMmap(
      _dataset_path + Constant::kTrainSetFile, DataType::kI32,
      {meta[Constant::kMetaNumTrainSet]}, CPU(), "dataset.train_set");
  _dataset->test_set = Tensor::FromMmap(
      _dataset_path + Constant::kTestSetFile, DataType::kI32,
      {meta[Constant::kMetaNumTestSet]}, CPU(), "dataset.test_set");
  _dataset->valid_set = Tensor::FromMmap(
      _dataset_path + Constant::kValidSetFile, DataType::kI32,
      {meta[Constant::kMetaNumValidSet]}, CPU(), "dataset.valid_set");

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
