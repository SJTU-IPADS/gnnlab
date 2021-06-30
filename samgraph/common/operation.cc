#include "operation.h"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "common.h"
#include "constant.h"
#include "engine.h"
#include "logging.h"
#include "profiler.h"
#include "run_config.h"

namespace samgraph {
namespace common {

extern "C" {

void samgraph_config(const char *path, int run_arch, int sample_type,
                     int sampler_device_type, int sampler_device_id,
                     int trainer_device_type, int trainer_device_id,
                     size_t batch_size, int *fanout, size_t num_fanout,
                     size_t num_epoch, int cache_policy,
                     double cache_percentage) {
  RunConfig::dataset_path = path;
  RunConfig::run_arch = static_cast<RunArch>(run_arch);
  RunConfig::sample_type = static_cast<SampleType>(sample_type);
  RunConfig::fanout = std::vector<int>(fanout, fanout + num_fanout);
  RunConfig::batch_size = batch_size;
  RunConfig::num_epoch = num_epoch;
  RunConfig::sampler_ctx =
      Context{static_cast<DeviceType>(sampler_device_type), sampler_device_id};
  RunConfig::trainer_ctx =
      Context{static_cast<DeviceType>(trainer_device_type), trainer_device_id};
  RunConfig::cache_policy = static_cast<CachePolicy>(cache_policy);
  RunConfig::cache_percentage = cache_percentage;

  std::unordered_map<SampleType, std::string> sample2str = {
      {kKHop0, "KHop0"},
      {kKHop1, "KHop1"},
      {kWeightedKHop, "WeightedKHop"},
      {kRandomWalk, "RandomWalk"},
  };

  LOG(INFO) << "Use " << sample2str[RunConfig::sample_type]
            << " sampling algorithm";

  RunConfig::LoadConfigFromEnv();
}

void samgraph_init() {
  Engine::Create();
  Engine::Get()->Init();

  LOG(INFO) << "SamGraph has been initialied successfully";
}

void samgraph_start() {
  CHECK(Engine::Get()->IsInitialized() && !Engine::Get()->IsShutdown());
  if (RunConfig::option_profile_cuda) {
    CUDA_CALL(cudaProfilerStart());
  }

  Engine::Get()->Start();
  LOG(INFO) << "SamGraph has been started successfully";
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
  return Engine::Get()->GetGraphDataset()->feat->Shape().at(1);
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

size_t samgraph_get_graph_num_src(uint64_t key, int graph_id) {
  auto batch = Engine::Get()->GetGraphBatch();
  return batch->graphs[graph_id]->num_src;
}

size_t samgraph_get_graph_num_dst(uint64_t key, int graph_id) {
  auto batch = Engine::Get()->GetGraphBatch();
  return batch->graphs[graph_id]->num_dst;
}

size_t samgraph_get_graph_num_edge(uint64_t key, int graph_id) {
  auto batch = Engine::Get()->GetGraphBatch();
  return batch->graphs[graph_id]->num_edge;
}

void samgraph_shutdown() {
  Engine::Get()->Shutdown();
  if (RunConfig::option_profile_cuda) {
    CUDA_CALL(cudaProfilerStop());
  }
  LOG(INFO) << "SamGraph has been completely shutdown now";
}

void samgraph_log_step(uint64_t epoch, uint64_t step, int item, double val) {
  CHECK_LT(item, kNumLogStepItems);
  uint64_t key = Engine::Get()->GetBatchKey(epoch, step);
  Profiler::Get().LogStep(key, static_cast<LogStepItem>(item), val);
}

void samgraph_log_step_add(uint64_t epoch, uint64_t step, int item,
                           double val) {
  CHECK_LT(item, kNumLogStepItems);
  uint64_t key = Engine::Get()->GetBatchKey(epoch, step);
  Profiler::Get().LogStepAdd(key, static_cast<LogStepItem>(item), val);
}

void samgraph_log_epoch_add(uint64_t epoch, int item, double val) {
  CHECK_LT(item, kNumLogEpochItems);
  uint64_t key = Engine::Get()->GetBatchKey(epoch, 0);
  Profiler::Get().LogEpochAdd(key, static_cast<LogEpochItem>(item), val);
}

double samgraph_get_log_step_value(uint64_t epoch, uint64_t step, int item) {
  CHECK_LT(item, kNumLogStepItems);
  uint64_t key = Engine::Get()->GetBatchKey(epoch, step);
  return Profiler::Get().GetLogStepValue(key, static_cast<LogStepItem>(item));
}

double samgraph_get_log_epoch_value(uint64_t epoch, int item) {
  CHECK_LT(item, kNumLogEpochItems);
  return Profiler::Get().GetLogEpochValue(epoch,
                                          static_cast<LogEpochItem>(item));
}

void samgraph_report_step(uint64_t epoch, uint64_t step) {
  Profiler::Get().ReportStep(epoch, step);
}

void samgraph_report_step_average(uint64_t epoch, uint64_t step) {
  Profiler::Get().ReportStepAverage(epoch, step);
}

void samgraph_report_epoch(uint64_t epoch) {
  Profiler::Get().ReportEpoch(epoch);
}

void samgraph_report_epoch_average(uint64_t epoch) {
  Profiler::Get().ReportEpochAverage(epoch);
}

void samgraph_report_node_access() {
  if (RunConfig::option_log_node_access) {
    Profiler::Get().ReportNodeAccess();
  }
}
}
}  // namespace common
}  // namespace samgraph
