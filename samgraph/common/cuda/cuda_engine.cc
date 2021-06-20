#include "cuda_engine.h"

#include <chrono>
#include <cstdlib>
#include <numeric>

#include "../common.h"
#include "../constant.h"
#include "../device.h"
#include "../logging.h"
#include "../run_config.h"
#include "cuda_common.h"
#include "cuda_loops.h"

namespace samgraph {
namespace common {
namespace cuda {

GPUEngine::GPUEngine() {
  _initialize = false;
  _should_shutdown = false;
}

void GPUEngine::Init() {
  if (_initialize) {
    return;
  }

  _sampler_ctx = RunConfig::sampler_ctx;
  _trainer_ctx = RunConfig::trainer_ctx;
  _dataset_path = RunConfig::dataset_path;
  _batch_size = RunConfig::batch_size;
  _fanout = RunConfig::fanout;
  _num_epoch = RunConfig::num_epoch;
  _joined_thread_cnt = 0;

  // Check whether the ctx configuration is allowable
  ArchCheck();

  // Load the target graph data
  LoadGraphDataset();

  // Create CUDA streams
  _sample_stream = Device::Get(_sampler_ctx)->CreateStream(_sampler_ctx);
  _sampler_copy_stream = Device::Get(_sampler_ctx)->CreateStream(_sampler_ctx);
  _trainer_copy_stream = Device::Get(_trainer_ctx)->CreateStream(_trainer_ctx);

  Device::Get(_sampler_ctx)->StreamSync(_sampler_ctx, _sample_stream);
  Device::Get(_sampler_ctx)->StreamSync(_sampler_ctx, _sampler_copy_stream);
  Device::Get(_trainer_ctx)->StreamSync(_trainer_ctx, _trainer_copy_stream);

  _shuffler =
      new GPUShuffler(_dataset->train_set, _num_epoch, _batch_size, false);
  _num_step = _shuffler->NumStep();
  _graph_pool = new GraphPool(RunConfig::kPipelineDepth);

  _hashtable = new OrderedHashTable(
      PredictNumNodes(_batch_size, _fanout, _fanout.size()), _sampler_ctx,
      _sample_stream);

  if (RunConfig::UseGPUCache() && RunConfig::run_arch != kArch1) {
    _cache_manager = new GPUCacheManager(
        _sampler_ctx, _trainer_ctx, _dataset->feat->Data(),
        _dataset->feat->Type(), _dataset->feat->Shape()[1],
        static_cast<const IdType*>(_dataset->sorted_nodes_by_in_degree->Data()),
        _dataset->num_node, RunConfig::cache_percentage);
  } else {
    _cache_manager = nullptr;
  }

  // Create CUDA random states for sampling
  _random_states = new GPURandomStates(RunConfig::sample_type, _fanout,
                                       _batch_size, _sampler_ctx);

  // Create queues
  for (int i = 0; i < QueueNum; i++) {
    LOG(DEBUG) << "Create task queue" << i;
    _queues.push_back(new TaskQueue(RunConfig::kPipelineDepth));
  }

  _initialize = true;
}

void GPUEngine::Start() {
  std::vector<LoopFunction> func;
  switch (RunConfig::run_arch) {
    case kArch1:
      func = GetArch1Loops();
      break;
    case kArch2:
      func = GetArch2Loops();
      break;
    case kArch3:
      func = GetArch3Loops();
      break;
    default:
      // Not supported arch 0
      CHECK(0);
  }

  // Start background threads
  for (size_t i = 0; i < func.size(); i++) {
    _threads.push_back(new std::thread(func[i]));
  }
  LOG(DEBUG) << "Started " << func.size() << " background threads.";
}

void GPUEngine::Shutdown() {
  if (_should_shutdown) {
    return;
  }

  _should_shutdown = true;
  int total_thread_num = _threads.size();

  while (!IsAllThreadFinish(total_thread_num)) {
    // wait until all threads joined
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }

  for (size_t i = 0; i < _threads.size(); i++) {
    _threads[i]->join();
    delete _threads[i];
    _threads[i] = nullptr;
  }

  // free queue
  for (size_t i = 0; i < QueueNum; i++) {
    if (_queues[i]) {
      delete _queues[i];
      _queues[i] = nullptr;
    }
  }

  Device::Get(_sampler_ctx)->StreamSync(_sampler_ctx, _sample_stream);
  Device::Get(_sampler_ctx)->FreeStream(_sampler_ctx, _sample_stream);
  Device::Get(_sampler_ctx)->StreamSync(_sampler_ctx, _sampler_copy_stream);
  Device::Get(_sampler_ctx)->FreeStream(_sampler_ctx, _sampler_copy_stream);

  Device::Get(_trainer_ctx)->StreamSync(_trainer_ctx, _trainer_copy_stream);
  Device::Get(_trainer_ctx)->FreeStream(_trainer_ctx, _trainer_copy_stream);

  delete _dataset;
  delete _shuffler;
  delete _graph_pool;
  delete _random_states;

  if (_cache_manager != nullptr) {
    delete _cache_manager;
  }

  _dataset = nullptr;
  _shuffler = nullptr;
  _graph_pool = nullptr;
  _cache_manager = nullptr;
  _random_states = nullptr;

  _threads.clear();
  _joined_thread_cnt = 0;
  _initialize = false;
  _should_shutdown = false;
}

void GPUEngine::RunSampleOnce() {
  switch (RunConfig::run_arch) {
    case kArch1:
      RunArch1LoopsOnce();
      break;
    case kArch2:
      RunArch2LoopsOnce();
      break;
    case kArch3:
      RunArch3LoopsOnce();
      break;
    default:
      // Not supported arch 0
      CHECK(0);
  }
}

void GPUEngine::Report(uint64_t epoch, uint64_t step) {
  Engine::Report(epoch, step);
}

void GPUEngine::ArchCheck() {
  CHECK_EQ(_sampler_ctx.device_type, kGPU);
  CHECK_EQ(_trainer_ctx.device_type, kGPU);

  if (RunConfig::run_arch == kArch1 || RunConfig::run_arch == kArch2) {
    CHECK_EQ(_sampler_ctx.device_id, _trainer_ctx.device_id);
  } else if (RunConfig::run_arch == kArch3) {
    CHECK_NE(_sampler_ctx.device_id, _trainer_ctx.device_id);
  } else {
    CHECK(0);
  }
}

std::unordered_map<std::string, Context> GPUEngine::GetGraphFileCtx() {
  std::unordered_map<std::string, Context> ret;

  ret[Constant::kIndptrFile] = _sampler_ctx;
  ret[Constant::kIndicesFile] = _sampler_ctx;
  ret[Constant::kTrainSetFile] = CPU();
  ret[Constant::kTestSetFile] = CPU();
  ret[Constant::kValidSetFile] = CPU();
  ret[Constant::kProbTableFile] = _sampler_ctx;
  ret[Constant::kAliasTableFile] = _sampler_ctx;
  ret[Constant::kInDegreeFile] = MMAP();
  ret[Constant::kOutDegreeFile] = MMAP();
  ret[Constant::kSortedNodeByInDegreeFile] = MMAP();

  switch (RunConfig::run_arch) {
    case kArch1:
      ret[Constant::kFeatFile] = _sampler_ctx;
      ret[Constant::kLabelFile] = _sampler_ctx;
      break;
    case kArch2:
    case kArch3:
      ret[Constant::kFeatFile] = MMAP();
      ret[Constant::kLabelFile] = MMAP();
      break;
    default:
      CHECK(0);
  }

  return ret;
}

}  // namespace cuda
}  // namespace common
}  // namespace samgraph
