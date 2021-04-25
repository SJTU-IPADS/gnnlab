#include "cpu_engine.h"

#include "../constant.h"
#include "../device.h"
#include "../logging.h"
#include "../run_config.h"
#include "../timer.h"
#include "cpu_loops.h"
#include "cpu_parallel_hashtable.h"
#include "cpu_simple_hashtable.h"

namespace samgraph {
namespace common {
namespace cpu {

CPUEngine::CPUEngine() {
  _initialize = false;
  _should_shutdown = false;
}

void CPUEngine::Init() {
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
  _work_stream = static_cast<cudaStream_t>(
      Device::Get(_trainer_ctx)->CreateStream(_trainer_ctx));
  Device::Get(_trainer_ctx)->StreamSync(_trainer_ctx, _work_stream);

  _extractor = new Extractor();
  _permutator =
      new CPUPermutator(_dataset->train_set, _num_epoch, _batch_size, false);
  _num_step = _permutator->NumStep();
  _graph_pool = new GraphPool(RunConfig::kPipelineDepth);

  switch (RunConfig::cpu_hashtable_type) {
    case kSimple:
      _hash_table = new SimpleHashTable(_dataset->num_node);
      break;
    case kParallel:
      _hash_table = new ParallelHashTable(_dataset->num_node);
      break;
    case kOptimized:
      CHECK(0);
      break;
    default:
      CHECK(0);
  }

  _initialize = true;
}

void CPUEngine::Start() {
  std::vector<LoopFunction> func;

  func.push_back(CPUSampleLoop);

  // Start background threads
  for (size_t i = 0; i < func.size(); i++) {
    _threads.push_back(new std::thread(func[i]));
  }
  LOG(DEBUG) << "Started " << func.size() << " background threads.";
}

void CPUEngine::Shutdown() {
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

  Device::Get(_trainer_ctx)->StreamSync(_trainer_ctx, _work_stream);
  Device::Get(_trainer_ctx)->FreeStream(_trainer_ctx, _work_stream);

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

void CPUEngine::RunSampleOnce() { RunCPUSampleLoopOnce(); }

void CPUEngine::Report(uint64_t epoch, uint64_t step) {
  Engine::Report(epoch, step);
}

}  // namespace cpu
}  // namespace common
}  // namespace samgraph
