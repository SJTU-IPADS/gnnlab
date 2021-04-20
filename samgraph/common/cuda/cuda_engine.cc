#include "cuda_engine.h"

#include <chrono>
#include <cstdlib>
#include <numeric>

#include "../common.h"
#include "../config.h"
#include "../logging.h"
#include "cuda_common.h"
#include "cuda_loops.h"

namespace samgraph {
namespace common {
namespace cuda {

GpuEngine::GpuEngine() {
  _initialize = false;
  _should_shutdown = false;
}

void GpuEngine::Init(std::string dataset_path, int sample_device,
                     int train_device, size_t batch_size,
                     std::vector<int> fanout, size_t num_epoch) {
  if (_initialize) {
    return;
  }

  CHECK_GT(sample_device, CPU_DEVICE_ID);
  CHECK_GT(train_device, CPU_DEVICE_ID);

  _sample_device = sample_device;
  _train_device = train_device;
  _dataset_path = dataset_path;
  _batch_size = batch_size;
  _fanout = fanout;
  _num_epoch = num_epoch;
  _joined_thread_cnt = 0;

  // Load the target graph data
  LoadGraphDataset();

  // Create CUDA streams
  CUDA_CALL(cudaSetDevice(_sample_device));
  CUDA_CALL(cudaStreamCreateWithFlags(&_sample_stream, cudaStreamNonBlocking));
  CUDA_CALL(cudaStreamCreateWithFlags(&_copy_stream, cudaStreamNonBlocking));
  CUDA_CALL(cudaStreamSynchronize(_sample_stream));
  CUDA_CALL(cudaStreamSynchronize(_copy_stream));

  _extractor = new Extractor();
  _permutator =
      new CudaPermutator(_dataset->train_set, _num_epoch, _batch_size, false);
  _num_step = _permutator->NumStep();
  _graph_pool = new GraphPool(Config::kPipelineThreshold);

  size_t predict_node_num =
      _batch_size + _batch_size * std::accumulate(_fanout.begin(),
                                                  _fanout.end(), 1ul,
                                                  std::multiplies<size_t>());
  _hashtable =
      new OrderedHashTable(predict_node_num, sample_device, _sample_stream, 3);

  // Create queues
  for (int i = 0; i < QueueNum; i++) {
    LOG(DEBUG) << "Create task queue" << i;
    _queues.push_back(new TaskQueue(Config::kPipelineThreshold, nullptr));
  }

  _initialize = true;
}

void GpuEngine::Start() {
  std::vector<LoopFunction> func;

  func.push_back(GpuSampleLoop);
  func.push_back(DataCopyLoop);

  // Start background threads
  for (size_t i = 0; i < func.size(); i++) {
    _threads.push_back(new std::thread(func[i]));
  }
  LOG(DEBUG) << "Started " << func.size() << " background threads.";
}

void GpuEngine::Shutdown() {
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

  CUDA_CALL(cudaStreamSynchronize(_sample_stream));
  CUDA_CALL(cudaStreamSynchronize(_copy_stream));
  CUDA_CALL(cudaStreamDestroy(_sample_stream));
  CUDA_CALL(cudaStreamDestroy(_copy_stream));

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

void GpuEngine::RunSampleOnce() {
  RunGpuSampleLoopOnce();
  RunDataCopyLoopOnce();
}

}  // namespace cuda
}  // namespace common
}  // namespace samgraph