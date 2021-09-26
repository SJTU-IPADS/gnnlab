#ifndef SAMGRAPH_DIST_ENGINE_H
#define SAMGRAPH_DIST_ENGINE_H

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "../common.h"
#include "../engine.h"
#include "../logging.h"
#include "../graph_pool.h"
#include "../task_queue.h"
#include "../cuda/cuda_common.h"
#include "../cuda/cuda_frequency_hashmap.h"
#include "../cuda/cuda_hashtable.h"
#include "../cuda/cuda_random_states.h"
#include "dist_shuffler.h"
#include "dist_cache_manager.h"

namespace samgraph {
namespace common {
namespace dist {

enum class DistType {Sample = 0, Extract, Default};

class DistEngine : public Engine {
 public:
  DistEngine();

  void Init() override;
  void Start() override;
  void Shutdown() override;
  void RunSampleOnce() override;
  void SampleInit(int worker_id, Context ctx);
  void TrainInit(int worker_id, Context ctx);
  /**
   * @param count: the total times to loop extract
   */
  void StartExtract(int count);

  // XXX: decide CPU or GPU to shuffling, sampling and id remapping
  DistShuffler* GetShuffler() { return static_cast<DistShuffler*>(_shuffler); }
  TaskQueue* GetTaskQueue(cuda::QueueType qt) { return _queues[qt]; }
  cuda::OrderedHashTable* GetHashtable() { return _hashtable; }
  cuda::GPURandomStates* GetRandomStates() { return _random_states; }
  DistCacheManager* GetCacheManager() { return _cache_manager; }
  cuda::FrequencyHashmap* GetFrequencyHashmap() { return _frequency_hashmap; }
  IdType *GetCacheHashtable() { return _cache_hashtable; }

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
  // set the feat and label if null
  void TrainDataLoad();
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
  // Frequency hashmap
  cuda::FrequencyHashmap* _frequency_hashmap;
  // vertices cache hash table
  IdType *_cache_hashtable;

  void ArchCheck() override;
  std::unordered_map<std::string, Context> GetGraphFileCtx() override;
  // Dist type: Sample or Extract
  DistType _dist_type;
};

}  // namespace dist
}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_DIST_ENGINE_H
