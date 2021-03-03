#ifndef SAMGRAPH_ENGINE_H
#define SAMGRAPH_ENGINE_H

#include <string>
#include <vector>
#include <thread>

#include <cuda_runtime.h>

#include "common.h"
#include "logging.h"
#include "task_queue.h"

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

  static void CreateTaskQueue(QueueType queueType);
  static void LoadGraphDataset();

  static cudaStream_t *GetSampleStream();
  static cudaStream_t *GetIdCopyHost2DeviceStream();
  static cudaStream_t *GetGraphCopyDevice2DeviceStream();
  static cudaStream_t *GetIdCopyDevice2HostStream();
  static cudaStream_t *GetFeatureCopyHost2DeviceStream();

  static void ReportThreadFinish() { joined_thread_cnt.fetch_add(1); }
  static bool IsAllThreadFinish(int total_thread_num);
  static std::atomic_int joined_thread_cnt;

 private:
  // whether the server is initialized
  static bool _initialize;
  // the engine is going to be shutdowned
  static bool _should_shutdown;
  // sampling engine device
  static int _sample_device;
  // training device
  static int _train_device;
  // dataset path
  static std::string _dataset_path;
  // global graph dataset
  static SamGraphDataset* _dataset;
  // sampling batch size
  static int _batch_size;
  // fanout data
  static std::vector<int> _fanout;
  // sampling epoch
  static int _num_epoch;
  // task queue
  static volatile SamGraphTaskQueue* _queues[QueueNum];
  static std::mutex _queues_mutex[QueueNum];
  static std::vector<std::thread*> _threads;
  // cuda streams
  static cudaStream_t* _sample_stream;
  static cudaStream_t* _id_copy_host2device_stream;
  static cudaStream_t* _graph_copy_device2device_stream;
  static cudaStream_t* _id_copy_device2host_stream;
  static cudaStream_t* _feat_copy_host2device_stream;
};

} // namespace common
} // namespace samgraph

#endif // SAMGRAPH_ENGINE_H