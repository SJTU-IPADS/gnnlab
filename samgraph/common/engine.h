#ifndef SAMGRAPH_ENGINE_H
#define SAMGRAPH_ENGINE_H

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <thread>

#include "common.h"
#include "constant.h"
#include "graph_pool.h"
#include "run_config.h"
#include "partition.h"

namespace samgraph {
namespace common {

class Engine {
 public:
  virtual void Init() = 0;
  virtual void Start() = 0;
  virtual void Shutdown() = 0;
  virtual void RunSampleOnce() = 0;

  std::vector<size_t> GetFanout() { return _fanout; }
  size_t NumEpoch() { return _num_epoch; }
  size_t NumStep() { return _num_step; }
  size_t NumLocalStep() { return _num_local_step; }

  inline uint64_t GetBatchKey(uint64_t epoch, uint64_t step) {
    return epoch * _num_step + step;
  }
  inline uint64_t GetEpochFromKey(uint64_t key) { return key / _num_step; };
  inline uint64_t GetStepFromKey(uint64_t key) { return key % _num_step; }

  bool ShouldShutdown() { return _should_shutdown; }
  bool IsInitialized() { return _initialize; }
  bool IsShutdown() { return _should_shutdown; }

  Context GetSamplerCtx() { return _sampler_ctx; }
  Context GetTrainerCtx() { return _trainer_ctx; }

  const Dataset* GetGraphDataset() { return _dataset; }

  GraphPool* GetGraphPool() { return _graph_pool; }
  std::shared_ptr<GraphBatch> GetGraphBatch() { return _graph_batch; };
  void SetGraphBatch(std::shared_ptr<GraphBatch> batch) {
    _graph_batch = batch;
  }
  void WaitBarrier() {
    while (inner_counter > outer_counter) {
      std::this_thread::sleep_for(std::chrono::nanoseconds(100));
    }
  }
  void ForwardBarrier();
  void ForwardInnerBarrier();

  void ReportThreadFinish() { _joined_thread_cnt.fetch_add(1); }
  virtual void ExamineDataset() {}

  static void Create();
  static Engine* Get() { return _engine; }

 protected:
  // Whether the server is initialized
  volatile bool _initialize;
  // The engine is going to be shutdowned
  volatile bool _should_shutdown;
  // Sampling engine device
  Context _sampler_ctx;
  // Training device
  Context _trainer_ctx;
  // Dataset path
  std::string _dataset_path;
  // Global graph dataset
  Dataset* _dataset;
  // Sampling batch size
  size_t _batch_size;
  // Fanout data
  std::vector<size_t> _fanout;
  // Sampling epoch
  size_t _num_epoch;
  // Number of steps per epoch
  size_t _num_step;
  // Number of steps per epoch in the view of the current worker
  size_t _num_local_step;
  // Ready graph batch pool
  GraphPool* _graph_pool;
  // Current graph batch
  std::shared_ptr<GraphBatch> _graph_batch;
  // Current graph batch
  std::atomic_int _joined_thread_cnt;

#ifdef PARTITION_TEST
  std::unique_ptr<Partition> partition;
#endif

  virtual void ArchCheck() = 0;
  virtual std::unordered_map<std::string, Context> GetGraphFileCtx() = 0;

  void LoadGraphDataset();
  void SortUMDatasetBy(const IdType* order);
  bool IsAllThreadFinish(int total_thread_num);

  volatile int inner_counter = 0;
  volatile int outer_counter = 0;

  static Engine* _engine;
};

}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_ENGINE_H