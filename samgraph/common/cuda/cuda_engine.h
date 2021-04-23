#ifndef SAMGRAPH_CUDA_ENGINE_H
#define SAMGRAPH_CUDA_ENGINE_H

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "../common.h"
#include "../engine.h"
#include "../extractor.h"
#include "../graph_pool.h"
#include "../logging.h"
#include "../ready_table.h"
#include "../task_queue.h"
#include "cuda_common.h"
#include "cuda_hashtable.h"
#include "cuda_permutator.h"

namespace samgraph {
namespace common {
namespace cuda {

class GpuEngine : public Engine {
 public:
  GpuEngine();

  void Init() override;
  void Start() override;
  void Shutdown() override;
  void RunSampleOnce() override;

  CudaPermutator* GetPermutator() { return _permutator; }
  Extractor* GetExtractor() { return _extractor; }
  TaskQueue* GetTaskQueue(QueueType qt) { return _queues[qt]; }
  OrderedHashTable* GetHashtable() { return _hashtable; }

  StreamHandle GetSampleStream() { return _sample_stream; }
  StreamHandle GetCopyStream() { return _copy_stream; }

  static GpuEngine* Get() { return dynamic_cast<GpuEngine*>(Engine::_engine); }

 private:
  // Task queue
  std::vector<TaskQueue*> _queues;
  std::vector<std::thread*> _threads;
  // Cuda streams on sample device
  StreamHandle _sample_stream;
  StreamHandle _copy_stream;
  // Random node batch genrator
  CudaPermutator* _permutator;
  // CPU Extractor
  Extractor* _extractor;
  // Hash table
  OrderedHashTable* _hashtable;
};

}  // namespace cuda
}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_CUDA_ENGINE_H