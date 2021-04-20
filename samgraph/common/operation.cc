#include "operation.h"

#include <cuda_profiler_api.h>

#include <string>
#include <vector>

#include "cpu/cpu_loops.h"
#include "engine.h"
#include "logging.h"
#include "macros.h"
#include "profiler.h"

namespace samgraph {
namespace common {

extern "C" {

void samgraph_init(const char *path, int sample_device, int train_device,
                   size_t batch_size, int *fanout, size_t num_fanout,
                   size_t num_epoch) {
  Engine::Create(sample_device);
  Engine::Get()->Init(path, sample_device, train_device, batch_size,
                      std::vector<int>(fanout, fanout + num_fanout), num_epoch);
  LOG(DEBUG) << "SamGraph has been initialied successfully";
#if PROFILE_CUDA_KERNELS
  CUDA_CALL(cudaProfilerStart());
#endif
  return;
}

void samgraph_start() {
  CHECK(Engine::Get()->IsInitialized() && !Engine::Get()->IsShutdown());
  Engine::Get()->Start();
  LOG(DEBUG) << "SamGraph has been started successfully";
}

size_t samgraph_num_epoch() {
  CHECK(Engine::Get()->IsInitialized() && !Engine::Get()->IsShutdown());
  return Engine::Get()->NumEpoch();
}

size_t samgraph_steps_per_epoch() {
  CHECK(Engine::Get()->IsInitialized() && !Engine::Get()->IsShutdown());
  return Engine::Get()->NumStep();
}

size_t samgraph_num_class() {
  CHECK(Engine::Get()->IsInitialized() && !Engine::Get()->IsShutdown());
  return Engine::Get()->GetGraphDataset()->num_class;
}

size_t samgraph_feat_dim() {
  CHECK(Engine::Get()->IsInitialized() && !Engine::Get()->IsShutdown());
  return Engine::Get()->GetGraphDataset()->feat->shape().at(1);
}

uint64_t samgraph_get_next_batch(uint64_t epoch, uint64_t step) {
  CHECK(Engine::Get()->IsInitialized() && !Engine::Get()->IsShutdown());

  uint64_t key = Engine::Get()->GetBatchKey(epoch, step);
  LOG(DEBUG) << "samgraph_get_next_batch encodeKey with epoch " << epoch
             << " step " << step << " and key " << key;
  auto graph = Engine::Get()->GetGraphPool()->GetGraphBatch(key);

  LOG(DEBUG) << "Get next batch with key " << key;
  Engine::Get()->SetGraphBatch(graph);

  return key;
}

void samgraph_sample_once() { Engine::Get()->RunSampleOnce(); }

size_t samgraph_get_graph_num_row(uint64_t key, int graph_id) {
  auto batch = Engine::Get()->GetGraphBatch();
  return batch->graphs[graph_id]->num_row;
}

size_t samgraph_get_graph_num_col(uint64_t key, int graph_id) {
  auto batch = Engine::Get()->GetGraphBatch();
  return batch->graphs[graph_id]->num_column;
}

size_t samgraph_get_graph_num_edge(uint64_t key, int graph_id) {
  auto batch = Engine::Get()->GetGraphBatch();
  return batch->graphs[graph_id]->num_edge;
}

void samgraph_shutdown() {
  Engine::Get()->Shutdown();
  LOG(DEBUG) << "SamGraph has been completely shutdown now";
#if PROFILE_CUDA_KERNELS
  CUDA_CALL(cudaProfilerStop());
#endif
  return;
}

void samgraph_profiler_report(uint64_t epoch, uint64_t step) {
  uint64_t key = Engine::Get()->GetBatchKey(epoch, step);
  Profiler::Get()->Report(key);
}
}
}  // namespace common
}  // namespace samgraph
