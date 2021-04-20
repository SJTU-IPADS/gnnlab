#ifndef SAMGRAPH_ENGINE_H
#define SAMGRAPH_ENGINE_H

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "common.h"
#include "graph_pool.h"

namespace samgraph {
namespace common {

typedef void (*LoopFunction)();

class Engine {
 public:
  virtual void Init(std::string dataset_path, int sample_device,
                    int train_device, size_t batch_size,
                    std::vector<int> fanout, size_t num_epoch) = 0;
  virtual void Start() = 0;
  virtual void Shutdown() = 0;
  virtual void RunSampleOnce() = 0;

  std::vector<int> GetFanout() { return _fanout; }
  size_t NumEpoch() { return _num_epoch; }
  size_t NumStep() { return _num_step; }

  uint64_t GetBatchKey(uint64_t epoch, uint64_t step) {
    return epoch * _num_step + step;
  }
  uint64_t GetEpochFromKey(uint64_t key) { return key / _num_step; };
  uint64_t GetStepFromKey(uint64_t key) { return key % _num_step; }

  bool ShouldShutdown() { return _should_shutdown; }
  bool IsInitialized() { return _initialize; }
  bool IsShutdown() { return _should_shutdown; }

  int GetSampleDevice() { return _sample_device; }
  int GetTrainDevice() { return _train_device; }

  const Dataset* GetGraphDataset() { return _dataset; }

  GraphPool* GetGraphPool() { return _graph_pool; }
  std::shared_ptr<GraphBatch> GetGraphBatch() { return _graph_batch; };
  void SetGraphBatch(std::shared_ptr<GraphBatch> batch) {
    _graph_batch = batch;
  }

  void ReportThreadFinish() { _joined_thread_cnt.fetch_add(1); }

  static void Create(int device);
  static Engine* Get() { return _engine; }

 protected:
  // Whether the server is initialized
  bool _initialize;
  // The engine is going to be shutdowned
  bool _should_shutdown;
  // Sampling engine device
  int _sample_device;
  // Training device
  int _train_device;
  // Dataset path
  std::string _dataset_path;
  // Global graph dataset
  Dataset* _dataset;
  // Sampling batch size
  size_t _batch_size;
  // Fanout data
  std::vector<int> _fanout;
  // Sampling epoch
  size_t _num_epoch;
  // Number of steps per epoch
  size_t _num_step;
  // Ready graph batch pool
  GraphPool* _graph_pool;
  // Current graph batch
  std::shared_ptr<GraphBatch> _graph_batch;
  // Current graph batch
  std::atomic_int _joined_thread_cnt;

  void LoadGraphDataset();
  bool IsAllThreadFinish(int total_thread_num);

  static Engine* _engine;
};

}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_ENGINE_H