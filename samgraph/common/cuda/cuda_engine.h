/*
 * Copyright 2022 Institute of Parallel and Distributed Systems, Shanghai Jiao Tong University
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

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
#include "../task_queue.h"
#include "cuda_cache_manager.h"
#include "cuda_common.h"
#include "cuda_frequency_hashmap.h"
#include "cuda_hashtable.h"
#include "cuda_random_states.h"
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

  Shuffler* GetShuffler() { return _shuffler; }
  TaskQueue* GetTaskQueue(QueueType qt) { return _queues[qt]; }
  OrderedHashTable* GetHashtable() { return _hashtable; }
  GPURandomStates* GetRandomStates() { return _random_states; }
  GPUCacheManager* GetCacheManager() { return _cache_manager; }
  GPUDynamicCacheManager* GetDynamicCacheManager() { return _dynamic_cache_manager; }
  FrequencyHashmap* GetFrequencyHashmap() { return _frequency_hashmap; }

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
  // Random node batch generator
  Shuffler* _shuffler;
  // Hash table
  OrderedHashTable* _hashtable;
  // CUDA random states
  GPURandomStates* _random_states;
  // Feature cache in GPU
  GPUCacheManager* _cache_manager;
  GPUDynamicCacheManager* _dynamic_cache_manager;
  // Frequency hashmap
  FrequencyHashmap* _frequency_hashmap;

  void ArchCheck() override;
  std::unordered_map<std::string, Context> GetGraphFileCtx() override;
};

}  // namespace cuda
}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_CUDA_ENGINE_H
