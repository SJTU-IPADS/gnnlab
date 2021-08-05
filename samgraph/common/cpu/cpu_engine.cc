#include "cpu_engine.h"

#include "../constant.h"
#include "../device.h"
#include "../logging.h"
#include "../run_config.h"
#include "../timer.h"
#include "cpu_hashtable0.h"
#include "cpu_hashtable1.h"
#include "cpu_hashtable2.h"
#include "cpu_loops.h"

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

  // Check whether the ctx configuration is allowable
  ArchCheck();

  // Load the target graph data
  LoadGraphDataset();

  // Create CUDA streams
  _work_stream = static_cast<cudaStream_t>(
      Device::Get(_trainer_ctx)->CreateStream(_trainer_ctx));
  Device::Get(_trainer_ctx)->StreamSync(_trainer_ctx, _work_stream);

  _shuffler =
      new CPUShuffler(_dataset->train_set, _num_epoch, _batch_size, false);
  _num_step = _shuffler->NumStep();
  _graph_pool = new GraphPool(RunConfig::max_copying_jobs);

  switch (RunConfig::cpu_hash_type) {
    case kCPUHash0:
      _hash_table = new CPUHashTable0(_dataset->num_node);
      break;
    case kCPUHash1:
      _hash_table = new CPUHashTable1(_dataset->num_node);
      break;
    case kCPUHash2:
      _hash_table = new CPUHashTable2(_dataset->num_node);
      break;
    default:
      CHECK(0);
  }

  LOG(INFO) << "CPU Engine uses type " << RunConfig::cpu_hash_type
            << " hashtable";

  _initialize = true;
}

void CPUEngine::Start() {
  std::vector<LoopFunction> func = GetArch0Loops();

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

  Device::Get(_trainer_ctx)->StreamSync(_trainer_ctx, _work_stream);
  Device::Get(_trainer_ctx)->FreeStream(_trainer_ctx, _work_stream);

  delete _dataset;
  delete _shuffler;
  delete _graph_pool;
  delete _hash_table;
  _dataset = nullptr;
  _shuffler = nullptr;
  _graph_pool = nullptr;
  _hash_table = nullptr;

  _threads.clear();
  _joined_thread_cnt = 0;
  _initialize = false;
  _should_shutdown = false;
}

void CPUEngine::RunSampleOnce() { RunArch0LoopsOnce(); }

void CPUEngine::ArchCheck() {
  CHECK_EQ(RunConfig::run_arch, kArch0);
  CHECK_EQ(_sampler_ctx.device_type, kCPU);
  CHECK_EQ(_trainer_ctx.device_type, kGPU);
}

std::unordered_map<std::string, Context> CPUEngine::GetGraphFileCtx() {
  std::unordered_map<std::string, Context> ret;

  ret[Constant::kIndptrFile] = MMAP();
  ret[Constant::kIndicesFile] = MMAP();
  ret[Constant::kFeatFile] = MMAP();
  ret[Constant::kLabelFile] = MMAP();
  ret[Constant::kTrainSetFile] = CPU();
  ret[Constant::kTestSetFile] = CPU();
  ret[Constant::kValidSetFile] = CPU();
  ret[Constant::kProbTableFile] = MMAP();
  ret[Constant::kAliasTableFile] = MMAP();
  ret[Constant::kInDegreeFile] = MMAP();
  ret[Constant::kOutDegreeFile] = MMAP();
  ret[Constant::kCacheByDegreeFile] = MMAP();
  ret[Constant::kCacheByHeuristicFile] = MMAP();

  return ret;
}

void CPUEngine::ExamineDataset() {
  IdType max_degree = 0;
  auto ds = GetGraphDataset();
  const IdType *indptr = static_cast<const IdType *>(ds->indptr->Data());
  // const IdType *indices = static_cast<const IdType *>(ds->indices->Data());
  // const IdType *in_degrees = static_cast<const IdType *>(ds->in_degrees->Data());
  // for (IdType i = 0; i < ds->num_node - 1; i++) {
  //   assert(indptr[i + 1] - indptr[i] == in_degrees[i]);
  // }
  for (IdType i = 0; i < ds->num_node-1; i++) {
    if (indptr[i + 1] - indptr[i] > max_degree) {
      max_degree = indptr[i + 1] - indptr[i];
    }
  }
  if (ds->num_edge - indptr[ds->num_node-1] > max_degree) {
    max_degree = ds->num_edge - indptr[ds->num_node-1];
  }
  LOG(ERROR) << "total nodes is " << ds->num_node;
  LOG(ERROR) << "max degree is " << max_degree;
}

}  // namespace cpu
}  // namespace common
}  // namespace samgraph
