#ifndef SAMGRAPH_CPU_ENGINE_H
#define SAMGRAPH_CPU_ENGINE_H

#include <cuda_runtime.h>

#include <thread>

#include "../cuda/cuda_cache_manager.h"
#include "../engine.h"
#include "../logging.h"
#include "cpu_hashtable.h"
#include "cpu_shuffler.h"

namespace samgraph {
namespace common {
namespace cpu {

class CPUEngine : public Engine {
 public:
  CPUEngine();

  void Init() override;
  void Start() override;
  void Shutdown() override;
  void RunSampleOnce() override;
  void ExamineDataset() override;

  CPUShuffler* GetShuffler() { return _shuffler; }
  cudaStream_t GetWorkStream() { return _work_stream; }
  CPUHashTable* GetHashTable() { return _hash_table; }
  cuda::GPUCacheManager* GetCacheManager() { return _cache_manager; }

  static CPUEngine* Get() { return dynamic_cast<CPUEngine*>(Engine::_engine); }

 private:
  // Task queue
  std::vector<std::thread*> _threads;

  cudaStream_t _work_stream;
  // Random node batch generator
  CPUShuffler* _shuffler;
  // Hash table
  CPUHashTable* _hash_table;
  // GPU cache manager
  cuda::GPUCacheManager* _cache_manager;

  void ArchCheck() override;
  std::unordered_map<std::string, Context> GetGraphFileCtx() override;
};

}  // namespace cpu
}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_CPU_ENGINE_H