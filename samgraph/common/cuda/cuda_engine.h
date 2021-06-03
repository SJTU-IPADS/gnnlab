#ifndef SAMGRAPH_CUDA_ENGINE_H
#define SAMGRAPH_CUDA_ENGINE_H

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "../common.h"
#include "../engine.h"
#include "../graph_pool.h"
#include "../logging.h"
#include "../task_queue.h"
#include "cuda_cache_manager.h"
#include "cuda_common.h"
#include "cuda_hashtable.h"
#include "cuda_shuffler.h"

namespace samgraph {
namespace common {
namespace cuda {

class GPUEngine : public Engine {
 public:
  GPUEngine();

  void Init() override;
  void Start() override;
  void Shutdown() override;
  void RunSampleOnce() override;
  void Report(uint64_t epoch, uint64_t step) override;

  GPUShuffler* GetShuffler() { return _shuffler; }
  TaskQueue* GetTaskQueue(QueueType qt) { return _queues[qt]; }
  OrderedHashTable* GetHashtable() { return _hashtable; }
  GPUCacheManager* GetCacheManager() { return _cache_manager; }

  StreamHandle GetSampleStream() { return _sample_stream; }
  StreamHandle GetSamplerCopyStream() { return _sampler_copy_stream; }
  StreamHandle GetTrainerCopyStream() { return _trainer_copy_stream; }

  static GPUEngine* Get() { return dynamic_cast<GPUEngine*>(Engine::_engine); }

 private:
  // Task queue
  std::vector<TaskQueue*> _queues;
  std::vector<std::thread*> _threads;
  // Cuda streams on sample device
  StreamHandle _sample_stream;
  StreamHandle _sampler_copy_stream;
  StreamHandle _trainer_copy_stream;
  // Random node batch genrator
  GPUShuffler* _shuffler;
  // Hash table
  OrderedHashTable* _hashtable;
  // Feature cache in GPU
  GPUCacheManager* _cache_manager;
};

}  // namespace cuda
}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_CUDA_ENGINE_H