#include "dist_engine.h"

#include <chrono>
#include <cstdlib>
#include <numeric>

#include "../constant.h"
#include "../device.h"
#include "../logging.h"
#include "../run_config.h"
#include "../timer.h"
#include "../cuda/cuda_common.h"
#include "../cpu/cpu_engine.h"
#include "dist_loops.h"

// TODO: decide CPU or GPU to shuffling, sampling and id remapping
/*
#include "cpu_hashtable0.h"
#include "cpu_hashtable1.h"
#include "cpu_hashtable2.h"
#include "cpu_loops.h"
*/

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
  _sample_stream = nullptr;
  _sampler_copy_stream = nullptr;
  _trainer_copy_stream = nullptr;

  // Check whether the ctx configuration is allowable
  DistEngine::ArchCheck();

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

void DistEngine::SampleInit(int device_type, int device_id) {
  if (_initialize) {
    LOG(FATAL) << "DistEngine already initialized!";
    return;
  }
  RunConfig::sampler_ctx = Context{static_cast<DeviceType>(device_type), device_id};
  _sampler_ctx = RunConfig::sampler_ctx;
  if (_sampler_ctx.device_type == kGPU) {
    _sample_stream = Device::Get(_sampler_ctx)->CreateStream(_sampler_ctx);
    // XXX: if remove value _sampler_copy_stream ?
    _sampler_copy_stream = Device::Get(_sampler_ctx)->CreateStream(_sampler_ctx);

    Device::Get(_sampler_ctx)->StreamSync(_sampler_ctx, _sample_stream);
    Device::Get(_sampler_ctx)->StreamSync(_sampler_ctx, _sampler_copy_stream);
  }
  // batch results set
  _graph_pool = new GraphPool(RunConfig::max_copying_jobs);

  SampleDataCopy(_sampler_ctx, _sample_stream);

  _shuffler = nullptr;
  switch(device_type) {
    case kCPU:
      _shuffler = new CPUShuffler(_dataset->train_set,
          _num_epoch, _batch_size, false);
      break;
    case kGPU:
      _shuffler = new cuda::GPUShuffler(_dataset->train_set,
          _num_epoch, _batch_size, false);
      break;
    default:
        LOG(FATAL) << "shuffler does not support device_type: "
                   << device_type;
  }
  _num_step = _shuffler->NumStep();
  // XXX: map the _hash_table to difference device
  //       _hashtable only support GPU device
  _hashtable = new cuda::OrderedHashTable(
      PredictNumNodes(_batch_size, _fanout, _fanout.size()), _sampler_ctx);

  // TODO: cache needs to support
  //       _trainer_ctx is not initialized
  if (RunConfig::UseGPUCache()) {
    _cache_manager = new cuda::GPUCacheManager(
        _sampler_ctx, _trainer_ctx, _dataset->feat->Data(),
        _dataset->feat->Type(), _dataset->feat->Shape()[1],
        static_cast<const IdType*>(_dataset->ranking_nodes->Data()),
        _dataset->num_node, RunConfig::cache_percentage);
  } else {
    _cache_manager = nullptr;
  }

  // Create CUDA random states for sampling
  _random_states = new cuda::GPURandomStates(RunConfig::sample_type, _fanout,
                                       _batch_size, _sampler_ctx);

  if (RunConfig::sample_type == kRandomWalk) {
    size_t max_nodes =
        PredictNumNodes(_batch_size, _fanout, _fanout.size() - 1);
    size_t edges_per_node =
        RunConfig::num_random_walk * RunConfig::random_walk_length;
    _frequency_hashmap =
        new cuda::FrequencyHashmap(max_nodes, edges_per_node, _sampler_ctx);
  } else {
    _frequency_hashmap = nullptr;
  }

  // Create queues
  // XXX: what is the usage of value _queues ?
  //      the differences between _queues and _graph_pool ?
  for (int i = 0; i < cuda::QueueNum; i++) {
    LOG(DEBUG) << "Create task queue" << i;
    _queues.push_back(new TaskQueue(RunConfig::max_sampling_jobs));
  }
  _graph_pool = new GraphPool(RunConfig::max_copying_jobs);

  _initialize = true;
}

void DistEngine::TrainInit(int device_type, int device_id) {
  if (_initialize) {
    LOG(FATAL) << "DistEngine already initialized!";
    return;
  }
  RunConfig::trainer_ctx = Context{static_cast<DeviceType>(device_type), device_id};
  _trainer_ctx = RunConfig::trainer_ctx;

  // Create CUDA streams
  // XXX: create cuda streams that training needs
  //       only support GPU sampling
  _trainer_copy_stream = Device::Get(_trainer_ctx)->CreateStream(_trainer_ctx);
  Device::Get(_trainer_ctx)->StreamSync(_trainer_ctx, _trainer_copy_stream);
  // next code is for CPU sampling
  /*
  _work_stream = static_cast<cudaStream_t>(
      Device::Get(_trainer_ctx)->CreateStream(_trainer_ctx));
  Device::Get(_trainer_ctx)->StreamSync(_trainer_ctx, _work_stream);
  */

  // results pool
  _graph_pool = new GraphPool(RunConfig::max_copying_jobs);

  _initialize = true;
}


void DistEngine::Start() {
  LOG(FATAL) << "DistEngine needs not implement the Start function!!!";
}

void DistEngine::Shutdown() {
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
  for (size_t i = 0; i < cuda::QueueNum; i++) {
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

  if (_frequency_hashmap != nullptr) {
    delete _frequency_hashmap;
  }

  _dataset = nullptr;
  _shuffler = nullptr;
  _graph_pool = nullptr;
  _cache_manager = nullptr;
  _random_states = nullptr;
  _frequency_hashmap = nullptr;

  _threads.clear();
  _joined_thread_cnt = 0;
  _initialize = false;
  _should_shutdown = false;
}

// TODO: implement it!
//       and split the sampling and extracting
void DistEngine::RunSampleOnce() {
  switch (RunConfig::run_arch) {
    case kArch5:
      RunArch5LoopsOnce();
      break;
    default:
      CHECK(0);
  }
  LOG(DEBUG) << "RunSampleOnce finished.";
}

void DistEngine::ArchCheck() {
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
