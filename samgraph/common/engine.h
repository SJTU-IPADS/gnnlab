#ifndef SAMGRAPH_ENGINE_H
#define SAMGRAPH_ENGINE_H

#include <string>
#include <vector>
#include <thread>

#include <cuda_runtime.h>

#include "common.h"
#include "logging.h"
#include "task_queue.h"
#include "random_permutation.h"
#include "graph_pool.h"

namespace samgraph {
namespace common {

typedef void (*LoopFunction)();

class SamGraphEngine {
 public:
  static void Init(std::string dataset_path, int sample_device, int train_device,
                   int batch_size, std::vector<int> fanout, int num_epoch);
  static void Start(const std::vector<LoopFunction> &func);
  static bool ShouldShutdown() { return _should_shutdown; }
  static void Shutdown();

  static std::vector<int> GetFanout() { return _fanout; }

  static RandomPermutation* GetRandomPermutation() { return _permutation; }

  static TaskQueue* GetTaskQueue(QueueType queueType) { return (TaskQueue*)_queues[queueType]; }
  static void CreateTaskQueue(QueueType queueType);
  static void LoadGraphDataset();
  static const SamGraphDataset* GetGraphDataset() { return _dataset; }

  static int GetSampleDevice() { return _sample_device; }
  static int GetTrainDevice() { return _train_device; }

  static cudaStream_t *GetSampleStream() { return _sample_stream; }
  static cudaStream_t *GetIdCopyHost2DeviceStream() { return _id_copy_host2device_stream; }
  static cudaStream_t *GetGraphCopyDevice2DeviceStream() { return _graph_copy_device2device_stream; }
  static cudaStream_t *GetIdCopyDevice2HostStream() { return _id_copy_device2host_stream; }
  static cudaStream_t *GetFeatureCopyHost2DeviceStream() {  return _feat_copy_host2device_stream; }

  static void ReportThreadFinish() { joined_thread_cnt.fetch_add(1); }
  static bool IsAllThreadFinish(int total_thread_num);
  static std::atomic_int joined_thread_cnt;

 private:
  // Whether the server is initialized
  static bool _initialize;
  // The engine is going to be shutdowned
  static bool _should_shutdown;
  // Sampling engine device
  static int _sample_device;
  // Training device
  static int _train_device;
  // Dataset path
  static std::string _dataset_path;
  // Global graph dataset
  static SamGraphDataset* _dataset;
  // Sampling batch size
  static int _batch_size;
  // Fanout data
  static std::vector<int> _fanout;
  // Sampling epoch
  static int _num_epoch;
  // Task queue
  static volatile SamGraphTaskQueue* _queues[QueueNum];
  static std::vector<std::thread*> _threads;
  // Cuda streams
  static cudaStream_t* _sample_stream;
  static cudaStream_t* _id_copy_host2device_stream;
  static cudaStream_t* _graph_copy_device2device_stream;
  static cudaStream_t* _id_copy_device2host_stream;
  static cudaStream_t* _feat_copy_host2device_stream;
  // Random node batch genrator
  static RandomPermutation *_permutation;
  // Ready graph batch pool
  static GraphPool *_graph_pool;
};

} // namespace common
} // namespace samgraph

#endif // SAMGRAPH_ENGINE_H