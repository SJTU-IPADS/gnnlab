#ifndef SAMGRAPH_DIST_ENGINE_H
#define SAMGRAPH_DIST_ENGINE_H

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "../common.h"
#include "../cuda/cuda_cache_manager.h"
#include "../cuda/cuda_common.h"
#include "../cuda/cuda_frequency_hashmap.h"
#include "../cuda/cuda_hashtable.h"
#include "../cuda/cuda_random_states.h"
#include "../engine.h"
#include "../graph_pool.h"
#include "../logging.h"
#include "../task_queue.h"
#include "dist_cache_manager.h"

namespace samgraph {
namespace common {
namespace dist {

class DistSharedBarrier {
 public:
  DistSharedBarrier(int count);
  ~ DistSharedBarrier() {
    munmap(_barrier_ptr, sizeof(pthread_barrier_t));
  }
  void Wait();
 private:
  pthread_barrier_t* _barrier_ptr;
};

enum class DistType {Sample = 0, Extract, Switch, Default};

class DistEngine : public Engine {
 public:
  DistEngine();

  void Init() override;
  void Start() override;
  void Shutdown() override;
  void RunSampleOnce() override;
  void SampleInit(int worker_id, Context ctx);
  void TrainInit(int worker_id, Context ctx, DistType dist_type);
  /**
   * @param count: the total times to loop extract
   */
  void StartExtract(int count);

  // XXX: decide CPU or GPU to shuffling, sampling and id remapping
  Shuffler* GetShuffler() { return _shuffler; }
  TaskQueue* GetTaskQueue(cuda::QueueType qt) { return _queues[qt]; }
  cuda::OrderedHashTable* GetHashtable() { return _hashtable; }
  cuda::GPURandomStates* GetRandomStates() { return _random_states; }
  DistCacheManager* GetCacheManager() { return _cache_manager; }
  cuda::GPUCacheManager* GetGPUCacheManager() { return _gpu_cache_manager; }
  cuda::FrequencyHashmap* GetFrequencyHashmap() { return _frequency_hashmap; }
  IdType *GetCacheHashtable() { return _cache_hashtable; }
  DistType GetDistType() { return _dist_type; }

  StreamHandle GetSampleStream() { return _sample_stream; }
  StreamHandle GetSamplerCopyStream() { return _sampler_copy_stream; }
  StreamHandle GetTrainerCopyStream() { return _trainer_copy_stream; }

  static DistEngine* Get() { return dynamic_cast<DistEngine*>(Engine::_engine); }

 private:
  // Copy data sampling needed for subprocess
  void SampleDataCopy(Context sampler_ctx, StreamHandle stream);
  void SampleCacheTableInit();
  // Copy data training needed for subprocess
  void TrainDataCopy(Context trainer_ctx, StreamHandle stream);
  // Task queue
  std::vector<TaskQueue*> _queues;
  std::vector<std::thread*> _threads;

  // Cuda streams on sample device
  StreamHandle _sample_stream;
  StreamHandle _sampler_copy_stream;
  StreamHandle _trainer_copy_stream;
  // Random node batch generator
  Shuffler* _shuffler;
  // Hash table
  cuda::OrderedHashTable* _hashtable;
  // CUDA random states
  cuda::GPURandomStates* _random_states;
  // Feature cache in GPU
  DistCacheManager* _cache_manager;
  // Cuda Cache Manager
  cuda::GPUCacheManager* _gpu_cache_manager;
  // Frequency hashmap
  cuda::FrequencyHashmap* _frequency_hashmap;
  // vertices cache hash table
  IdType *_cache_hashtable;

  void ArchCheck() override;
  std::unordered_map<std::string, Context> GetGraphFileCtx() override;
  // Dist type: Sample or Extract
  DistType _dist_type;

  MessageTaskQueue *_memory_queue;
  DistSharedBarrier *_sampler_barrier;
};

}  // namespace dist
}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_DIST_ENGINE_H
