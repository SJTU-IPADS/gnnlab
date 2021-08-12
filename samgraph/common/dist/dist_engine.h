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
#include "../cuda/cuda_cache_manager.h"
#include "../cuda/cuda_common.h"
#include "../cuda/cuda_frequency_hashmap.h"
#include "../cuda/cuda_hashtable.h"
#include "../cuda/cuda_random_states.h"
#include "../cuda/cuda_shuffler.h"

namespace samgraph {
namespace common {
namespace dist {

using namespace cuda;

class DistEngine : public Engine {
 public:
  DistEngine();

  void Init() override;
  void Start() override;
  void Shutdown() override;
  void RunSampleOnce() override;
  void SampleInit(int device_type, int device_id);
  void TrainInit(int device_type, int device_id);

  // TODO: decide CPU or GPU to shuffling, sampling and id remapping
  GPUShuffler* GetShuffler() { return static_cast<GPUShuffler*>(_shuffler); }
  TaskQueue* GetTaskQueue(QueueType qt) { return _queues[qt]; }
  OrderedHashTable* GetHashtable() { return _hashtable; }
  GPURandomStates* GetRandomStates() { return _random_states; }
  GPUCacheManager* GetCacheManager() { return _cache_manager; }
  FrequencyHashmap* GetFrequencyHashmap() { return _frequency_hashmap; }

  StreamHandle GetSampleStream() { return _sample_stream; }
  StreamHandle GetSamplerCopyStream() { return _sampler_copy_stream; }
  StreamHandle GetTrainerCopyStream() { return _trainer_copy_stream; }

  static DistEngine* Get() { return dynamic_cast<DistEngine*>(Engine::_engine); }

 private:
  // Copy data sampling needed for subprocess
  void SampleDataCopy(Context sampler_ctx, StreamHandle stream);
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
  OrderedHashTable* _hashtable;
  // CUDA random states
  GPURandomStates* _random_states;
  // Feature cache in GPU
  GPUCacheManager* _cache_manager;
  // Frequency hashmap
  FrequencyHashmap* _frequency_hashmap;

  void ArchCheck() override;
  std::unordered_map<std::string, Context> GetGraphFileCtx() override;
};

}  // namespace dist
}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_DIST_ENGINE_H
