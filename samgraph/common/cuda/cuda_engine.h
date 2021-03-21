#ifndef SAMGRAPH_CUDA_ENGINE_H
#define SAMGRAPH_CUDA_ENGINE_H

#include <string>
#include <vector>
#include <thread>
#include <memory>
#include <atomic>

#include <cuda_runtime.h>

#include "../common.h"
#include "../logging.h"
#include "../task_queue.h"
#include "../random_permutation.h"
#include "../graph_pool.h"
#include "../ready_table.h"
#include "../engine.h"
#include "cpu_extractor.h"

namespace samgraph {
namespace common {
namespace cuda {

class SamGraphCudaEngine : public SamGraphEngine {
 public:
  SamGraphCudaEngine();

  void Init(std::string dataset_path, int sample_device, int train_device,
                   size_t batch_size, std::vector<int> fanout, int num_epoch) override;
  void Start() override;
  void Shutdown() override;

  RandomPermutation* GetRandomPermutation() { return _permutation; }
  CpuExtractor* GetCpuExtractor() { return _cpu_extractor; }
  SamGraphTaskQueue* GetTaskQueue(CudaQueueType queueType) { return _queues[queueType]; }
  ReadyTable *GetSubmitTable() { return _submit_table; }

  cudaStream_t* GetSampleStream() { return _sample_stream; }
  cudaStream_t* GetIdCopyHost2DeviceStream() { return _id_copy_host2device_stream; }
  cudaStream_t* GetGraphCopyDevice2DeviceStream() { return _graph_copy_device2device_stream; }
  cudaStream_t* GetIdCopyDevice2HostStream() { return _id_copy_device2host_stream; }
  cudaStream_t* GetFeatureCopyHost2DeviceStream() {  return _feat_copy_host2device_stream; }

  static inline SamGraphCudaEngine *Get() { return dynamic_cast<SamGraphCudaEngine *>(SamGraphEngine::_engine); }

 private:
  // Task queue
  std::vector<SamGraphTaskQueue*> _queues;
  std::vector<std::thread*> _threads;
  // Cuda streams on sample device
  cudaStream_t* _sample_stream;
  cudaStream_t* _id_copy_host2device_stream;
  cudaStream_t* _graph_copy_device2device_stream;
  cudaStream_t* _id_copy_device2host_stream;
  // Cuda streams on train device
  cudaStream_t* _feat_copy_host2device_stream;
  
  // Random node batch genrator
  RandomPermutation* _permutation;
  // Ready table
  ReadyTable* _submit_table;
  // CPU Extractor
  CpuExtractor* _cpu_extractor;

  bool IsAllThreadFinish(int total_thread_num);
};

} // namespace cuda
} // namespace common
} // namespace samgraph

#endif // SAMGRAPH_CUDA_ENGINE_H