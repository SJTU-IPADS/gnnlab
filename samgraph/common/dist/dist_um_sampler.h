#ifndef SAMGRAPH_DIST_UM_SAMPLER_H
#define SAMGRAPH_DIST_UM_SAMPLER_H

#include <semaphore.h>

#include "../common.h"
#include "../device.h"
#include "../logging.h"
#include "../engine.h"
#include "../cuda/cuda_engine.h"
#include "../cuda/cuda_hashtable.h"
#include "../cuda/cuda_random_states.h"

// tiny engine for um sampler, maintain sampler ctx, etc

namespace samgraph {
namespace common {
namespace dist {

class DistUMSampler {
public:
    DistUMSampler(Dataset &dataset, IdType sampler_id);
    ~DistUMSampler();
    Device* GetDevice() { return Device::Get(_sampler_ctx); }
    Context Ctx() { return _sampler_ctx; };
    StreamHandle SampleStream() { return  _sample_stream; }
    StreamHandle CopyStream() { return _sampler_copy_stream; }
    sem_t* WorkerSem() { return &_sem; }

    Shuffler* GetShuffler() { return _shuffler; }
    cuda::OrderedHashTable* GetHashTable() { return _hashtable; }
    IdType* GetCacheHashTable() { return _cache_hashtable; }
    cuda::GPURandomStates* GetGPURandomStates() { return _random_states; }
    cuda::FrequencyHashmap* GetFrequencyHashmap() { return _frequency_hashmap; }
    GraphPool* GetGraphPool() { return _graph_pool; }
    std::thread::id WorkerId();

    void SyncSampler();
    void ReleaseSem();
    void AcquireSem();
    void CacheTableInit(const IdType* cpu_hashtb);
    void CreateWorker(std::function<bool(sem_t*)> sample_function, sem_t*  sem);


private:
    IdType _sampler_id;
    std::thread* _work_thread;
    sem_t _sem;

    Context _sampler_ctx;
    StreamHandle _sample_stream;
    StreamHandle _sampler_copy_stream;

    Dataset& _global_dataset;

    Shuffler* _shuffler;
    cuda::OrderedHashTable* _hashtable;
    cuda::GPURandomStates* _random_states;
    cuda::FrequencyHashmap* _frequency_hashmap;
    GraphPool* _graph_pool;

    IdType* _cache_hashtable;
};
    
} // namespace dist
} // namespace common
} // namespace samgraph



#endif