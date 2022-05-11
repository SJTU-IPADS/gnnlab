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
#include "dist_um_sampler.h"
#include "collaborative_cache_manager.h"

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
  void UMSampleInit(int num_workers);
  /**
   * @param count: the total times to loop extract
   */
  void StartExtract(int count);

  // XXX: decide CPU or GPU to shuffling, sampling and id remapping
  Shuffler* GetShuffler() { return _shuffler; }
  TaskQueue* GetTaskQueue(cuda::QueueType qt) { return _queues[qt]; }
  cuda::OrderedHashTable* GetHashtable() { 
    if (RunConfig::run_arch == RunArch::kArch9) {
      LOG(FATAL) << "arch9 should not use this function";
    }
    return _hashtable; 
  }
  cuda::GPURandomStates* GetRandomStates() {
    if (RunConfig::run_arch == RunArch::kArch9) {
      LOG(FATAL) << "arch9 should not use this function";
    }
    return _random_states; 
  }
#ifdef SAMGRAPH_LEGACY_CACHE_ENABLE
  DistCacheManager* GetCacheManager() { return _cache_manager; }
  cuda::GPUCacheManager* GetGPUCacheManager() { return _gpu_cache_manager; }
#endif
#ifdef SAMGRAPH_COLL_CACHE_ENABLE
  CollCacheManager* GetCollCacheManager() { return _coll_cache_manager; }
  CollCacheManager* GetCollLabelManager() { return _coll_label_manager; }
#endif
  cuda::FrequencyHashmap* GetFrequencyHashmap() { return _frequency_hashmap; }
#ifdef SAMGRAPH_LEGACY_CACHE_ENABLE
  IdType *GetCacheHashtable() { return _cache_hashtable; }
#endif
  DistType GetDistType() { return _dist_type; }

  GraphPool* GetGraphPool() override {
    if (RunConfig::run_arch == RunArch::kArch9 && _dist_type == DistType::Sample) {
      LOG(WARNING) << WARNING_PREFIX << "arch9 sampler should not use this function";
    }
    return _graph_pool;
  }

  StreamHandle GetSampleStream() { 
    if (RunConfig::run_arch == RunArch::kArch9) {
      LOG(WARNING) << WARNING_PREFIX << "arch9: get sample stream";
    }
    return _sample_stream; 
  }
  StreamHandle GetSamplerCopyStream() {
    if (RunConfig::run_arch == RunArch::kArch9) {
      LOG(WARNING) << WARNING_PREFIX << "arch9: get sampler copy stream";
    }
    return _sampler_copy_stream; 
  }
  StreamHandle GetTrainerCopyStream() { return _trainer_copy_stream; }

  std::vector<DistUMSampler*>& GetUMSamplers() { return _um_samplers; } 
  // because sampler is a process originally, 
  // there will be a lot modification if pass sampler pointer to sample function
  DistUMSampler* GetUMSamplerByTid(std::thread::id tid);
  DistSharedBarrier* GetTrainerBarrier() { return _trainer_barrier; }

  static DistEngine* Get() { return dynamic_cast<DistEngine*>(Engine::_engine); }

 private:
  // Copy data sampling needed for subprocess
  void SampleDataCopy(Context sampler_ctx, StreamHandle stream);
  void UMSampleLoadGraph();
#ifdef SAMGRAPH_LEGACY_CACHE_ENABLE
  void SampleCacheTableInit();
#endif
  void UMSampleCacheTableInit();
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
#ifdef SAMGRAPH_LEGACY_CACHE_ENABLE
  // Feature cache in GPU
  DistCacheManager* _cache_manager;
  // Cuda Cache Manager
  cuda::GPUCacheManager* _gpu_cache_manager;
#endif
#ifdef SAMGRAPH_COLL_CACHE_ENABLE
  // Collaborative cache manager
  CollCacheManager* _coll_cache_manager;
  CollCacheManager* _coll_label_manager;
#endif
  // Frequency hashmap
  cuda::FrequencyHashmap* _frequency_hashmap;
#ifdef SAMGRAPH_LEGACY_CACHE_ENABLE
  // vertices cache hash table
  IdType *_cache_hashtable;
#endif

  void ArchCheck() override;
  std::unordered_map<std::string, Context> GetGraphFileCtx() override;
  // Dist type: Sample or Extract
  DistType _dist_type;

  MessageTaskQueue *_memory_queue;
  DistSharedBarrier *_sampler_barrier;

  std::vector<DistUMSampler*> _um_samplers;
  DistSharedBarrier *_trainer_barrier;
};

}  // namespace dist
}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_DIST_ENGINE_H
