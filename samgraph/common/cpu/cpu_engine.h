#ifndef SAMGRAPH_CPU_ENGINE_H
#define SAMGRAPH_CPU_ENGINE_H

#include <cuda_runtime.h>

#include <thread>

#include "../engine.h"
#include "../extractor.h"
#include "../logging.h"
#include "cpu_hashtable.h"
#include "cpu_permutator.h"

namespace samgraph {
namespace common {
namespace cpu {

class CpuEngine : public Engine {
 public:
  CpuEngine();

  void Init() override;
  void Start() override;
  void Shutdown() override;
  void RunSampleOnce() override;

  CpuPermutator* GetPermutator() { return _permutator; }
  Extractor* GetExtractor() { return _extractor; }
  cudaStream_t GetWorkStream() { return _work_stream; }
  HashTable* GetHashTable() { return _hash_table; }

  static CpuEngine* Get() { return dynamic_cast<CpuEngine*>(Engine::_engine); }

 private:
  // Task queue
  std::vector<std::thread*> _threads;

  cudaStream_t _work_stream;

  // Random node batch genrator
  CpuPermutator* _permutator;
  // CPU Extractor
  Extractor* _extractor;
  // Hash table
  HashTable* _hash_table;
};

}  // namespace cpu
}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_CPU_ENGINE_H