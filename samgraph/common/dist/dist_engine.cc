#include "dist_engine.h"

#include "../constant.h"
#include "../device.h"
#include "../logging.h"
#include "../run_config.h"
#include "../timer.h"

// TODO: decide CPU or GPU to shuffling, sampling and id remapping
#include "cpu_hashtable0.h"
#include "cpu_hashtable1.h"
#include "cpu_hashtable2.h"
#include "cpu_loops.h"

namespace samgraph {
namespace common {
namespace dist {

DistEngine::DistEngine() {
  _initialize = false;
  _should_shutdown = false;
}

void DistEngine::Init() {
  if (_initialize) {
    return;
  }

  _dataset_path = RunConfig::dataset_path;
  _batch_size = RunConfig::batch_size;
  _fanout = RunConfig::fanout;
  _num_epoch = RunConfig::num_epoch;
  _joined_thread_cnt = 0;
  _sampler_stream = 0;

  // Check whether the ctx configuration is allowable
  ArchCheck();

  // Load the target graph data
  LoadGraphDataset();

  LOG(DEBUG) << "Finished pre-initialization";
}

void DistEngine::SampleDataCopy(Context sampler_ctx, StreamHandle stream) {
  _dataset->train_set = Tensor::CopyTo(_dataset->train_set, CPU(), stream);
  _dataset->valid_set = Tensor::CopyTo(_dataset->valid_set, CPU(), stream);
  _dataset->test_set = Tensor::CopyTo(_dataset->test_set, CPU(), stream);
  if (sampler_ctx.device_type == kGPU) {
    _dataset->indptr = Tensor::CopyTo(_dataset->indptr, sampler_ctx, stream);
    _dataset->indices = Tensor::CopyTo(_dataset->indices, sampler_ctx, stream);
    if (RunConfig::sample_type == kWeightedKHop) {
      _dataset->prob_table->Tensor::CopyTo(_dataset->prob_table, sampler_ctx, stream);
      _dataset->alias_table->Tensor::CopyTo(_dataset->prob_table, sampler_ctx, stream);
    }
  }
}

// TODO: add sample_init for sampling process
void DistEngine::SampleInit(int device_type, int device_id) {
  if (_initialize) {
    LOG(FATAL) << "DistEngine already initialized!";
    return;
  }
  RunConfig::sampler_ctx = Context{static_cast<DeviceType>(device_type), device_id};
  _sampler_ctx = RunConfig::sampler_ctx;
  if (_sampler_ctx.device_type == kGPU) {
    _sampler_stream = Device.Get(_sampler_ctx)->CreateStream(_sampler_ctx);
  }
  // batch results set
  _graph_pool = new GraphPool(RunConfig::max_copying_jobs);

  Device::Get(sampler_ctx)
  SampleDataCopy(_sampler_ctx, _sampler_stream);

  _shuffler = nullptr;
  switch(device_type) {
    case kCPU:
      _shuffler = new CPUShuffler(_dataset->train_set,
          _num_epoch, _batch_size, false);
      break;
    case kGPU:
      _shuffler = new GPUShuffler(_dataset->train_set,
          _num_epoch, _batch_size, false);
      break;
    default:
        LOG(FATAL) << "shuffler does not support device_type: "
                   << device_type;
  }
  _num_step = _shuffler->NumStep();
  // TODO: map the _hash_table to difference device

  _initialize = true;
}

// TODO: add train_init for extracting and training process
void DistEngine::TrainInit(int device_type, int device_id) {
  if (_initialize) {
    LOG(FATAL) << "DistEngine already initialized!";
    return;
  }
  RunConfig::trainer_ctx = Context{static_cast<DeviceType>(device_type), device_id};
  _trainer_ctx = RunConfig::trainer_ctx;

  // Create CUDA streams
  _work_stream = static_cast<cudaStream_t>(
      Device::Get(_trainer_ctx)->CreateStream(_trainer_ctx));
  Device::Get(_trainer_ctx)->StreamSync(_trainer_ctx, _work_stream);

  _initialize = true;
}


void DistEngine::Start() {
  LOG(FATAL) << "DistEngine needs not implement the Start function!!!";
}

void DistEngine::Shutdown() {
  LOG(FATAL) << "DistEngine needs not implement the Shutdown function!!!";
}

// TODO: implement it!
//       and split the sampling and extracting
void DistEngine::RunSampleOnce() {

  LOG(DEBUG) << "RunSampleOnce finished.";
}

void CPUEngine::ArchCheck() {
  CHECK_EQ(RunConfig::run_arch, kArch5);
}

std::unordered_map<std::string, Context> DistEngine::GetGraphFileCtx() {
  std::unordered_map<std::string, Context> ret;

  ret[Constant::kIndptrFile] = MMAP();
  ret[Constant::kIndicesFile] = MMAP();
  ret[Constant::kFeatFile] = MMAP();
  ret[Constant::kLabelFile] = MMAP();
  ret[Constant::kTrainSetFile] = MMAP();
  ret[Constant::kTestSetFile] = MMAP();
  ret[Constant::kValidSetFile] = MMAP();
  ret[Constant::kProbTableFile] = MMAP();
  ret[Constant::kAliasTableFile] = MMAP();
  ret[Constant::kInDegreeFile] = MMAP();
  ret[Constant::kOutDegreeFile] = MMAP();
  ret[Constant::kCacheByDegreeFile] = MMAP();
  ret[Constant::kCacheByHeuristicFile] = MMAP();

  return ret;
}

}  // namespace dist
}  // namespace common
}  // namespace samgraph
