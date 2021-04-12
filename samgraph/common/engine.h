#ifndef SAMGRAPH_ENGINE_H
#define SAMGRAPH_ENGINE_H

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "common.h"
#include "graph_pool.h"

namespace samgraph {
namespace common {

typedef void (*LoopFunction)();

class SamGraphEngine {
 public:
  virtual void Init(std::string dataset_path, int sample_device,
                    int train_device, size_t batch_size,
                    std::vector<int> fanout, int num_epoch) = 0;
  virtual void Start() = 0;
  virtual void Shutdown() = 0;
  virtual void SampleOnce() = 0;

  std::vector<int> GetFanout() { return _fanout; }
  int GetNumEpoch() { return _num_epoch; }
  size_t GetNumStep() { return _num_step; }

  bool ShouldShutdown() { return _should_shutdown; }
  bool IsInitialized() { return _initialize; }
  bool IsShutdown() { return _should_shutdown; }

  int GetSampleDevice() { return _sample_device; }
  int GetTrainDevice() { return _train_device; }

  const SamGraphDataset* GetGraphDataset() { return _dataset; }
  GraphPool* GetGraphPool() { return _graph_pool; }
  std::shared_ptr<GraphBatch> GetGraphBatch() { return _cur_graph_batch; };
  void SetGraphBatch(std::shared_ptr<GraphBatch> batch) {
    _cur_graph_batch = batch;
  }

  void ReportThreadFinish() { _joined_thread_cnt.fetch_add(1); }

  // Singleton
  static void CreateEngine(int device);
  static inline SamGraphEngine* GetEngine() { return _engine; }

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
  SamGraphDataset* _dataset;
  // Sampling batch size
  size_t _batch_size;
  // Fanout data
  std::vector<int> _fanout;
  // Sampling epoch
  int _num_epoch;
  // Number of steps per epoch
  size_t _num_step;
  // Ready graph batch pool
  GraphPool* _graph_pool;
  // Current graph batch
  std::shared_ptr<GraphBatch> _cur_graph_batch;
  // Current graph batch
  std::atomic_int _joined_thread_cnt;

  void LoadGraphDataset();
  bool IsAllThreadFinish(int total_thread_num);

  static SamGraphEngine* _engine;
};

}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_ENGINE_H