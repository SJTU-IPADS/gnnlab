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

  // Load the target graph data
  LoadGraphDataset();

  // Create CUDA streams
  _sample_stream = Device::Get(_sampler_ctx)->CreateStream(_sampler_ctx);
  _copy_stream = Device::Get(_sampler_ctx)->CreateStream(_sampler_ctx);

  Device::Get(_sampler_ctx)->StreamSync(_sampler_ctx, _sample_stream);
  Device::Get(_sampler_ctx)->StreamSync(_sampler_ctx, _copy_stream);

  _extractor = new Extractor();
  _permutator =
      new CudaPermutator(_dataset->train_set, _num_epoch, _batch_size, false);
  _num_step = _permutator->NumStep();
  _graph_pool = new GraphPool(RunConfig::kPipelineDepth);

  size_t predict_node_num =
      _batch_size + _batch_size * std::accumulate(_fanout.begin(),
                                                  _fanout.end(), 1ul,
                                                  std::multiplies<size_t>());
  _hashtable =
      new OrderedHashTable(predict_node_num, _sampler_ctx, _sample_stream);

  // Create queues
  for (int i = 0; i < QueueNum; i++) {
    LOG(DEBUG) << "Create task queue" << i;
    _queues.push_back(new TaskQueue(RunConfig::kPipelineDepth, nullptr));
  }

  // Create CUDA random states for sampling
  _randomSeeder = new GPURandomSeeder();
  _randomSeeder->Init(_fanout, _sampler_ctx, _sample_stream, _batch_size);

  _initialize = true;
}

void GPUEngine::Start() {
  std::vector<LoopFunction> func;

  func.push_back(GPUSampleLoop);
  func.push_back(DataCopyLoop);

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

  delete _dataset;

  // free queue
  for (size_t i = 0; i < QueueNum; i++) {
    if (_queues[i]) {
      delete _queues[i];
      _queues[i] = nullptr;
    }
  }

  Device::Get(_sampler_ctx)->StreamSync(_sampler_ctx, _sample_stream);
  Device::Get(_sampler_ctx)->StreamSync(_sampler_ctx, _copy_stream);
  Device::Get(_sampler_ctx)->FreeStream(_sampler_ctx, _sample_stream);
  Device::Get(_sampler_ctx)->FreeStream(_sampler_ctx, _copy_stream);

  if (_permutator) {
    delete _permutator;
    _permutator = nullptr;
  }

  if (_graph_pool) {
    delete _graph_pool;
    _graph_pool = nullptr;
  }

  _threads.clear();
  _joined_thread_cnt = 0;
  _initialize = false;
  _should_shutdown = false;
}

void GPUEngine::RunSampleOnce() {
  RunGPUSampleLoopOnce();
  RunDataCopyLoopOnce();
}

void GPUEngine::Report(uint64_t epoch, uint64_t step) {
  Engine::Report(epoch, step);
}

}  // namespace cuda
}  // namespace common
}  // namespace samgraph
