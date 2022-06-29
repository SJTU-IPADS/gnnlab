#include <thread>

#include "dist_um_sampler.h"
#include "../logging.h"
#include "dist_engine.h"
#include "dist_shuffler.h"
#include "../run_config.h"
#include "../device.h"
#include "../memory_queue.h"
#include "../cuda/cuda_engine.h"

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
#define LOG_MEM_USAGE(LEVEL, sampler_ctx, title) \
  {\
    auto __sampler_ctx = sampler_ctx; \
    LOG(LEVEL) << "<" << (title)  << ">" \
               << " cuda usage: sampler " << __sampler_ctx << " " \
               << ToReadableSize(get_cuda_used(__sampler_ctx)); \
  }
}


DistUMSampler::DistUMSampler(Dataset& dataset, IdType sampler_id) 
    : _sampler_id(sampler_id),
    _work_thread(nullptr),
    _sampler_ctx(RunConfig::unified_memory_ctxes[sampler_id]),
    _global_dataset(dataset)
{
  CHECK(_sampler_ctx.device_type == DeviceType::kGPU);
  _sample_stream = Device::Get(_sampler_ctx)->CreateStream(_sampler_ctx);
  _sampler_copy_stream = Device::Get(_sampler_ctx)->CreateStream(_sampler_ctx);
  Device::Get(_sampler_ctx)->StreamSync(_sampler_ctx, _sample_stream);
  Device::Get(_sampler_ctx)->StreamSync(_sampler_ctx, _sampler_copy_stream);

  sem_init(&_sem, 0, 0);

  _shuffler = new DistShuffler(
    Tensor::CopyTo(dataset.train_set, CPU()), 
    DistEngine::Get()->NumEpoch(), DistEngine::Get()->GetBatchSize(), 
    sampler_id, RunConfig::num_sample_worker, RunConfig::num_train_worker, false);
  // if (_shuffler->NumStep() > mq_size) {
  //   LOG(FATAL) << "num step exceeds memory queue size!";
  // }

  _hashtable = new cuda::OrderedHashTable(
    PredictNumNodes(
      DistEngine::Get()->GetBatchSize(), 
      DistEngine::Get()->GetFanout(), 
      DistEngine::Get()->GetFanout().size()),
    _sampler_ctx, _sampler_copy_stream
  );
  LOG_MEM_USAGE(INFO, _sampler_ctx, "Init UM Sampler after create hash table");

  _random_states = new cuda::GPURandomStates(
    RunConfig::sample_type, 
    DistEngine::Get()->GetFanout(),
    DistEngine::Get()->GetBatchSize(),
    _sampler_ctx);
  LOG_MEM_USAGE(WARNING, _sampler_ctx, "Iint UM Sampler after create random states");

  if (RunConfig::sample_type == SampleType::kRandomWalk) {
    size_t max_nodes = PredictNumNodes(
      DistEngine::Get()->GetBatchSize(),
      DistEngine::Get()->GetFanout(),
      DistEngine::Get()->GetFanout().size() - 1
    );
    size_t edges_per_node = RunConfig::num_random_walk * RunConfig::random_walk_length;
    _frequency_hashmap = new cuda::FrequencyHashmap(max_nodes, edges_per_node, _sampler_ctx);
  } else {
    _frequency_hashmap = nullptr;
  }

  // create cache hash table later(after presample)
  _cache_hashtable = nullptr;

  _graph_pool = new GraphPool(RunConfig::max_copying_jobs);

  LOG_MEM_USAGE(WARNING, _sampler_ctx, "finish um sampler initialization") ;
} 


DistUMSampler::~DistUMSampler() {
  LOG(DEBUG) << "DistUMSampler deconstuctor";
  if (_work_thread != nullptr) {
    _work_thread->join();
  }
  delete _work_thread;
  LOG_MEM_USAGE(WARNING, Ctx(), "before destory um sampler");
  SyncSampler();
  GetDevice()->FreeStream(Ctx(), _sample_stream);
  GetDevice()->FreeStream(Ctx(), _sampler_copy_stream);
 
  sem_destroy(&_sem);
  
  delete _shuffler;
  delete _hashtable;
  delete _random_states;
  if (_frequency_hashmap) delete _frequency_hashmap;
  delete _graph_pool;
  
  if (_cache_hashtable != nullptr) {
    GetDevice()->FreeDataSpace(Ctx(), _cache_hashtable);
  }
  LOG(WARNING) << "um sampler(" << Ctx() << ") has been destoryed";
}


std::thread::id DistUMSampler::WorkerId() {
  if (_work_thread == nullptr) {
    return static_cast<std::thread::id>(0);
  } else {
    return _work_thread->get_id();
  }
}


void DistUMSampler::SyncSampler() {
  GetDevice()->StreamSync(Ctx(), _sample_stream);
  GetDevice()->StreamSync(Ctx(), _sampler_copy_stream);
}

void DistUMSampler::ReleaseSem() {
  sem_post(&_sem);
}

void DistUMSampler::AcquireSem() {
  sem_wait(&_sem);
}


void DistUMSampler::CacheTableInit(const IdType* cpu_hashtb) {
  size_t num_nodes = _global_dataset.num_node;
  _cache_hashtable = static_cast<IdType*>(GetDevice()->AllocDataSpace(
    Ctx(), sizeof(IdType) * num_nodes));
  
  GetDevice()->CopyDataFromTo(cpu_hashtb, 0, _cache_hashtable, 0, 
    sizeof(IdType) * num_nodes, CPU(), Ctx(), _sample_stream);
  
  LOG_MEM_USAGE(INFO, Ctx(), "after init cache table");
}

void DistUMSampler::CreateWorker(std::function<bool(sem_t*)> sample_function, sem_t* sem) {
  this->_work_thread = new std::thread(sample_function, sem);
}



} // namespace dist
} // namespace common
} // namespace samgraph
