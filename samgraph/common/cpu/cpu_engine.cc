#include "cpu_engine.h"

#include "../config.h"
#include "../logging.h"
#include "../timer.h"
#include "cpu_loops.h"

namespace samgraph {
namespace common {
namespace cpu {

CpuEngine::CpuEngine() {
  _initialize = false;
  _should_shutdown = false;
}

void CpuEngine::Init(std::string dataset_path, int sample_device,
                     int train_device, size_t batch_size,
                     std::vector<int> fanout, size_t num_epoch) {
  if (_initialize) {
    return;
  }

  CHECK_EQ(sample_device, CPU_DEVICE_ID);
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
  CUDA_CALL(cudaSetDevice(_train_device));
  CUDA_CALL(cudaStreamCreateWithFlags(&_work_stream, cudaStreamNonBlocking));
  CUDA_CALL(cudaStreamSynchronize(_work_stream));

  _extractor = new Extractor();
  _permutator =
      new CpuPermutator(_dataset->train_set, _num_epoch, _batch_size, false);
  _num_step = _permutator->NumStep();
  _graph_pool = new GraphPool(Config::kPipelineThreshold);
  _hash_table = new ParallelHashTable(_dataset->num_node);

  _initialize = true;
}

void CpuEngine::Start() {
  std::vector<LoopFunction> func;

  // func.push_back(CpuSampleLoop);

  // Start background threads
  for (size_t i = 0; i < func.size(); i++) {
    _threads.push_back(new std::thread(func[i]));
  }
  LOG(DEBUG) << "Started " << func.size() << " background threads.";
}

void CpuEngine::Shutdown() {
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

  if (_extractor) {
    delete _extractor;
    _extractor = nullptr;
  }

  delete _dataset;

  CUDA_CALL(cudaStreamSynchronize(_work_stream));
  CUDA_CALL(cudaStreamDestroy(_work_stream));

  if (_permutator) {
    delete _permutator;
    _permutator = nullptr;
  }

  if (_graph_pool) {
    delete _graph_pool;
    _graph_pool = nullptr;
  }

  if (_hash_table) {
    delete _hash_table;
    _hash_table = nullptr;
  }

  _threads.clear();
  _joined_thread_cnt = 0;
  _initialize = false;
  _should_shutdown = false;
}

void CpuEngine::RunSampleOnce() { RunCpuSampleLoopOnce(); }

}  // namespace cpu
}  // namespace common
}  // namespace samgraph
