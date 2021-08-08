#ifndef SAMGRAPH_DIST_ENGINE_H
#define SAMGRAPH_DIST_ENGINE_H

#include <cuda_runtime.h>

#include <thread>

#include "../engine.h"
#include "../logging.h"

// TODO: decide CPU or GPU to shuffling, sampling and id remapping
#include "cpu_hashtable.h"
#include "cpu_shuffler.h"

namespace samgraph {
namespace common {
namespace dist {

class DistEngine : public Engine {
 public:
  DistEngine();

  void Init() override;
  void Start() override;
  void Shutdown() override;
  void RunSampleOnce() override;
  void SampleInit(int device_type, int device_id);
  void TrainInit(int device_type, int device_id)

  // TODO: decide CPU or GPU to shuffling, sampling and id remapping
  CPUShuffler* GetShuffler() { return _shuffler; }
  cudaStream_t GetWorkStream() { return _work_stream; }
  CPUHashTable* GetHashTable() { return _hash_table; }

  static DistEngine* Get() { return dynamic_cast<DistEngine*>(Engine::_engine); }

 private:
  // Copy data sampling needed for subprocess
  void DistEngine::SampleDataCopy(Context sampler_ctx, StreamHandle stream);
  // Task queue
  std::vector<std::thread*> _threads;

  cudaStream_t _sampler_stream;
  cudaStream_t _work_stream;
  // Random node batch generator
  CPUShuffler* _shuffler;
  // Hash table
  CPUHashTable* _hash_table;

  void ArchCheck() override;
  std::unordered_map<std::string, Context> GetGraphFileCtx() override;
};

}  // namespace dist
}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_DIST_ENGINE_H
