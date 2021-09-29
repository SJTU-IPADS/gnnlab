#include "dist_engine.h"

#include <semaphore.h>
#include <stddef.h>
#include <chrono>
#include <cstdlib>
#include <numeric>
#include <iterator>
#include <sstream>
#include <string>
#include <unordered_map>

#include "../constant.h"
#include "../device.h"
#include "../logging.h"
#include "../run_config.h"
#include "../timer.h"
#include "../cuda/cuda_common.h"
#include "../cpu/cpu_engine.h"
#include "dist_loops.h"
#include "pre_sampler.h"
#include "../profiler.h"

// XXX: decide CPU or GPU to shuffling, sampling and id remapping
/*
#include "cpu_hashtable0.h"
#include "cpu_hashtable1.h"
#include "cpu_hashtable2.h"
#include "cpu_loops.h"
*/

namespace samgraph {
namespace common {
namespace dist {

namespace {
size_t get_cuda_used(Context ctx) {
  size_t free, used;
  cudaSetDevice(ctx.device_id);
  cudaMemGetInfo(&free, &used);
  return used - free;
}
#define LOG_MEM_USAGE(LEVEL, title, ctx) \
  {\
    auto _target_device = Device::Get(ctx); \
    LOG(LEVEL) << (title) << ", data alloc: " << ToReadableSize(_target_device->DataSize(ctx));\
    LOG(LEVEL) << (title) << ", workspace : " << ToReadableSize(_target_device->WorkspaceSize(ctx));\
    LOG(LEVEL) << (title) << ", total     : " << ToReadableSize(_target_device->TotalSize(ctx));\
    LOG(LEVEL) << "cuda usage: " << ToReadableSize(get_cuda_used(ctx));\
  }
} // namespace

DistSharedBarrier::DistSharedBarrier(int count) {
  size_t nbytes = sizeof(pthread_barrier_t);
  _barrier_ptr= static_cast<pthread_barrier_t*>(mmap(NULL, nbytes,
                      PROT_READ|PROT_WRITE, MAP_SHARED|MAP_ANONYMOUS, -1, 0));
  CHECK_NE(_barrier_ptr, MAP_FAILED);
  pthread_barrierattr_t attr;
  pthread_barrierattr_init(&attr);
  pthread_barrierattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
  pthread_barrier_init(_barrier_ptr, &attr, count);
}

void DistSharedBarrier::Wait() {
  int err = pthread_barrier_wait(_barrier_ptr);
  CHECK(err == PTHREAD_BARRIER_SERIAL_THREAD || err == 0);
}

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
  _dist_type = DistType::Default;
  _shuffler = nullptr;
  _random_states = nullptr;
  _cache_manager = nullptr;
  _frequency_hashmap = nullptr;
  _cache_hashtable = nullptr;

  // Check whether the ctx configuration is allowable
  DistEngine::ArchCheck();

  // Load the target graph data
  LoadGraphDataset();

  if (RunConfig::UseGPUCache()) {
    switch (RunConfig::cache_policy) {
      case kCacheByPreSampleStatic:
      case kCacheByPreSample: {
        size_t nbytes = sizeof(IdType) * _dataset->num_node;
        void *shared_ptr = (mmap(NULL, nbytes, PROT_READ|PROT_WRITE, MAP_SHARED|MAP_ANONYMOUS, -1, 0));
        _dataset->ranking_nodes = Tensor::FromBlob(
            shared_ptr, DataType::kI32, {_dataset->num_node}, Context{kMMAP, 0}, "ranking_nodes");
        break;
      }
      default: ;
    }
  }

  _memory_queue = new MessageTaskQueue(RunConfig::max_copying_jobs);
  LOG(DEBUG) << "create sampler barrier with " << RunConfig::num_sample_worker << " samplers";
  _sampler_barrier = new DistSharedBarrier(RunConfig::num_sample_worker);

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
  LOG(DEBUG) << "SampleDataCopy finished!";
}

void DistEngine::SampleCacheTableInit() {
  size_t num_nodes = _dataset->num_node;
  auto nodes = static_cast<const IdType*>(_dataset->ranking_nodes->Data());
  size_t num_cached_nodes = num_nodes *
                            (RunConfig::cache_percentage);
  auto cpu_device = Device::Get(CPU());
  auto sampler_gpu_device = Device::Get(_sampler_ctx);

  IdType *tmp_cpu_hashtable = static_cast<IdType *>(
      cpu_device->AllocDataSpace(CPU(), sizeof(IdType) * num_nodes));
  _cache_hashtable =
      static_cast<IdType *>(sampler_gpu_device->AllocDataSpace(
          _sampler_ctx, sizeof(IdType) * num_nodes));

  // 1. Initialize the cpu hashtable
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for (size_t i = 0; i < num_nodes; i++) {
    tmp_cpu_hashtable[i] = Constant::kEmptyKey;
  }

  // 2. Populate the cpu hashtable
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for (size_t i = 0; i < num_cached_nodes; i++) {
    tmp_cpu_hashtable[nodes[i]] = i;
  }

  // 3. Copy the cache from the cpu memory to gpu memory
  sampler_gpu_device->CopyDataFromTo(
      tmp_cpu_hashtable, 0, _cache_hashtable, 0,
      sizeof(IdType) * num_nodes, CPU(), _sampler_ctx);

  // 4. Free the cpu tmp cache data
  cpu_device->FreeDataSpace(CPU(), tmp_cpu_hashtable);

  LOG(INFO) << "GPU cache (policy: " << RunConfig::cache_policy
            << ") " << num_cached_nodes << " / " << num_nodes;
}

void DistEngine::SampleInit(int worker_id, Context ctx) {
  if (_initialize) {
    LOG(FATAL) << "DistEngine already initialized!";
    return;
  }
  _dist_type = DistType::Sample;
  RunConfig::sampler_ctx = ctx;
  _sampler_ctx = RunConfig::sampler_ctx;
  LOG_MEM_USAGE(WARNING, "before sample initialization", _sampler_ctx);
  _memory_queue->PinMemory();
  LOG_MEM_USAGE(WARNING, "before sample pin memory", _sampler_ctx);
  if (_sampler_ctx.device_type == kGPU) {
    _sample_stream = Device::Get(_sampler_ctx)->CreateStream(_sampler_ctx);
    // use sampler_ctx in task sending
    _sampler_copy_stream = Device::Get(_sampler_ctx)->CreateStream(_sampler_ctx);

    Device::Get(_sampler_ctx)->StreamSync(_sampler_ctx, _sample_stream);
    Device::Get(_sampler_ctx)->StreamSync(_sampler_ctx, _sampler_copy_stream);
  }
  LOG_MEM_USAGE(WARNING, "after create sampler stream", _sampler_ctx);

  SampleDataCopy(_sampler_ctx, _sample_stream);
  LOG_MEM_USAGE(WARNING, "after sample data copy", _sampler_ctx);

  _shuffler = nullptr;
  switch(ctx.device_type) {
    case kCPU:
      _shuffler = new CPUShuffler(_dataset->train_set,
          _num_epoch, _batch_size, false);
      break;
    case kGPU:
      _shuffler = new DistShuffler(_dataset->train_set,
          _num_epoch, _batch_size, worker_id, RunConfig::num_sample_worker, RunConfig::num_train_worker, false);
      break;
    default:
        LOG(FATAL) << "shuffler does not support device_type: "
                   << ctx.device_type;
  }
  _num_step = _shuffler->NumStep();
  LOG_MEM_USAGE(WARNING, "after create shuffler", _sampler_ctx);

  // XXX: map the _hash_table to difference device
  //       _hashtable only support GPU device
#ifndef SXN_NAIVE_HASHMAP
  _hashtable = new cuda::OrderedHashTable(
      PredictNumNodes(_batch_size, _fanout, _fanout.size()), _sampler_ctx);
#else
  _hashtable = new cuda::OrderedHashTable(
      _dataset->num_node, _sampler_ctx, 1);
#endif
  LOG_MEM_USAGE(WARNING, "after create hashtable", _sampler_ctx);

  // Create CUDA random states for sampling
  _random_states = new cuda::GPURandomStates(RunConfig::sample_type, _fanout,
                                       _batch_size, _sampler_ctx);
  LOG_MEM_USAGE(WARNING, "after create random states", _sampler_ctx);

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
  for (int i = 0; i < cuda::QueueNum; i++) {
    LOG(DEBUG) << "Create task queue" << i;
    if (static_cast<cuda::QueueType>(i) == cuda::kDataCopy) {
      _queues.push_back(_memory_queue);
    }
    else {
      _queues.push_back(new TaskQueue(RunConfig::max_sampling_jobs));
    }
  }
  // batch results set
  _graph_pool = new GraphPool(RunConfig::max_copying_jobs);

  double presample_time = 0;
  if (RunConfig::UseGPUCache()) {
    switch (RunConfig::cache_policy) {
      case kCacheByPreSampleStatic:
      case kCacheByPreSample: {
        Timer tp;
        std::cout << "before presample with worker " << worker_id << std::endl;
        if (worker_id == 0) {
          PreSampler::SetSingleton(new PreSampler(_dataset->train_set, RunConfig::batch_size));
          PreSampler::Get()->DoPreSample();
          PreSampler::Get()->GetRankNode(_dataset->ranking_nodes);
        }
        std::cout << "before barrier with worker " << worker_id << std::endl;
        _sampler_barrier->Wait();
        presample_time = tp.Passed();
        break;
      }
      default: ;
    }
    LOG_MEM_USAGE(WARNING, "after presample", _sampler_ctx);
    SampleCacheTableInit();
  }
  Profiler::Get().LogInit(kLogInitL1Presample, presample_time);

  LOG_MEM_USAGE(WARNING, "after finish sample initialization", _sampler_ctx);
  _initialize = true;
}

void DistEngine::TrainDataLoad() {
  if ((_dataset->feat != nullptr) && (_dataset->label != nullptr)) {
    return;
  }
  std::unordered_map<std::string, size_t> meta;
  std::unordered_map<std::string, Context> ctx_map = GetGraphFileCtx();

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
  CHECK(meta.count(Constant::kMetaFeatDim) > 0);
  if (_dataset->feat == nullptr) {
    _dataset->feat = Tensor::Empty(
        DataType::kF32,
        {meta[Constant::kMetaNumNode], meta[Constant::kMetaFeatDim]},
        ctx_map[Constant::kFeatFile], "dataset.feat");
  }
  if (_dataset->label == nullptr) {
    _dataset->label =
        Tensor::Empty(DataType::kI64, {meta[Constant::kMetaNumNode]},
                      ctx_map[Constant::kLabelFile], "dataset.label");
  }
}

void DistEngine::TrainDataCopy(Context trainer_ctx, StreamHandle stream) {
  _dataset->label = Tensor::CopyTo(_dataset->label, trainer_ctx, stream);
  LOG(DEBUG) << "TrainDataCopy finished!";
}

void DistEngine::TrainInit(int worker_id, Context ctx) {
  if (_initialize) {
    LOG(FATAL) << "DistEngine already initialized!";
    return;
  }
  _dist_type = DistType::Extract;
  RunConfig::trainer_ctx = ctx;
  _trainer_ctx = RunConfig::trainer_ctx;
  LOG_MEM_USAGE(WARNING, "before train initialization", _trainer_ctx);

  _memory_queue->PinMemory();
  LOG_MEM_USAGE(WARNING, "after pin memory", _trainer_ctx);

  TrainDataLoad();
  LOG_MEM_USAGE(WARNING, "after train data load", _trainer_ctx);

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

  _num_step = ((_dataset->train_set->Shape().front() + _batch_size - 1) / _batch_size);

  LOG_MEM_USAGE(WARNING, "after train create stream", _trainer_ctx);

  if (RunConfig::UseGPUCache()) {
    TrainDataCopy(_trainer_ctx, _trainer_copy_stream);
    // wait the presample
    // XXX: let the app ensure sampler initialization before trainer
    /*
    switch (RunConfig::cache_policy) {
      case kCacheByPreSampleStatic:
      case kCacheByPreSample: {
        break;
      }
      default: ;
    }
    */
    _cache_manager = new DistCacheManager(
        _trainer_ctx, _dataset->feat->Data(),
        _dataset->feat->Type(), _dataset->feat->Shape()[1],
        static_cast<const IdType*>(_dataset->ranking_nodes->Data()),
        _dataset->num_node, RunConfig::cache_percentage);
  } else {
    _cache_manager = nullptr;
  }
  LOG_MEM_USAGE(WARNING, "after train load cache", _trainer_ctx);

  // Create queues
  for (int i = 0; i < cuda::QueueNum; i++) {
    LOG(DEBUG) << "Create task queue" << i;
    if (static_cast<cuda::QueueType>(i) == cuda::kDataCopy) {
      _queues.push_back(_memory_queue);
    }
    else {
      _queues.push_back(new TaskQueue(RunConfig::max_sampling_jobs));
    }
  }
  // results pool
  _graph_pool = new GraphPool(RunConfig::max_copying_jobs);

  _initialize = true;
  LOG_MEM_USAGE(WARNING, "after train initialization", _trainer_ctx);
}


void DistEngine::Start() {
  LOG(FATAL) << "DistEngine needs not implement the Start function!!!";
}

/**
 * @param count: the total times to loop
 */
void DistEngine::StartExtract(int count) {
  ExtractFunction func;
  switch (RunConfig::run_arch) {
    case kArch5:
      func = GetArch5Loops();
      break;
    default:
      // Not supported arch 0
      CHECK(0);
  }

  // Start background threads
  _threads.push_back(new std::thread(func, count));
  LOG(DEBUG) << "Started a extracting background threads.";
}

void DistEngine::Shutdown() {
  if (_dist_type == DistType::Sample) {
    LOG_MEM_USAGE(WARNING, "sampler before shutdown", _sampler_ctx);
  }
  else if (_dist_type == DistType::Extract) {
    LOG_MEM_USAGE(WARNING, "trainer before shutdown", _trainer_ctx);
  }

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

  if (_dist_type == DistType::Sample) {
    Device::Get(_sampler_ctx)->StreamSync(_sampler_ctx, _sample_stream);
    Device::Get(_sampler_ctx)->FreeStream(_sampler_ctx, _sample_stream);
    Device::Get(_sampler_ctx)->StreamSync(_sampler_ctx, _sampler_copy_stream);
    Device::Get(_sampler_ctx)->FreeStream(_sampler_ctx, _sampler_copy_stream);
  }
  else if (_dist_type == DistType::Extract) {
    Device::Get(_trainer_ctx)->StreamSync(_trainer_ctx, _trainer_copy_stream);
    Device::Get(_trainer_ctx)->FreeStream(_trainer_ctx, _trainer_copy_stream);
  }
  else {
    LOG(FATAL) << "_dist_type is illegal!";
  }

  delete _dataset;
  delete _graph_pool;
  if (_shuffler != nullptr) {
    delete _shuffler;
  }
  if (_random_states != nullptr) {
    delete _random_states;
  }

  if (_cache_manager != nullptr) {
    delete _cache_manager;
  }

  if (_frequency_hashmap != nullptr) {
    delete _frequency_hashmap;
  }

  if (_cache_hashtable != nullptr) {
    Device::Get(_sampler_ctx)->FreeDataSpace(_sampler_ctx, _cache_hashtable);
  }

  if (_sampler_barrier != nullptr) {
    delete _sampler_barrier;
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
  LOG(INFO) << "DistEngine shutdown successfully!";
}

void DistEngine::RunSampleOnce() {
  switch (RunConfig::run_arch) {
    case kArch5:
      RunArch5LoopsOnce(_dist_type);
      break;
    default:
      CHECK(0);
  }
  LOG(DEBUG) << "RunSampleOnce finished.";
}

void DistEngine::ArchCheck() {
  CHECK_EQ(RunConfig::run_arch, kArch5);
  CHECK(!(RunConfig::UseGPUCache() && RunConfig::option_log_node_access));
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
  ret[Constant::kCacheByDegreeHopFile] = MMAP();
  ret[Constant::kCacheByFakeOptimalFile] = MMAP();

  return ret;
}

}  // namespace dist
}  // namespace common
}  // namespace samgraph
