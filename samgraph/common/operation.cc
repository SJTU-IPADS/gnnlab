#include "operation.h"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "common.h"
#include "constant.h"
#include "engine.h"
#include "logging.h"
#include "profiler.h"
#include "run_config.h"
#include "./dist/dist_engine.h"
#include "timer.h"

namespace samgraph {
namespace common {

extern "C" {

void samgraph_config(const char *path, int run_arch, int sample_type,
                     int sampler_device_type, int sampler_device_id,
                     int trainer_device_type, int trainer_device_id,
                     size_t batch_size, size_t num_epoch, int cache_policy,
                     double cache_percentage, size_t max_sampling_jobs,
                     size_t max_copying_jobs) {
  CHECK(!RunConfig::is_configured);
  RunConfig::dataset_path = path;
  RunConfig::run_arch = static_cast<RunArch>(run_arch);
  RunConfig::sample_type = static_cast<SampleType>(sample_type);
  RunConfig::batch_size = batch_size;
  RunConfig::num_epoch = num_epoch;
  RunConfig::sampler_ctx =
      Context{static_cast<DeviceType>(sampler_device_type), sampler_device_id};
  RunConfig::trainer_ctx =
      Context{static_cast<DeviceType>(trainer_device_type), trainer_device_id};
  RunConfig::cache_policy = static_cast<CachePolicy>(cache_policy);
  RunConfig::cache_percentage = cache_percentage;

  RunConfig::max_sampling_jobs = max_sampling_jobs;
  RunConfig::max_copying_jobs = max_copying_jobs;

  std::unordered_map<SampleType, std::string> sample2str = {
      {kKHop0, "KHop0"},
      {kKHop1, "KHop1"},
      {kWeightedKHop, "WeightedKHop"},
      {kRandomWalk, "RandomWalk"},
      {kWeightedKHopPrefix, "WeightedKHopPrefix"},
  };

  LOG(INFO) << "Use " << sample2str[RunConfig::sample_type]
            << " sampling algorithm";

  RunConfig::LoadConfigFromEnv();

  RunConfig::is_configured = true;
}

void samgraph_config_khop(size_t *fanout, size_t num_fanout) {
  CHECK(!RunConfig::is_khop_configured &&
        !RunConfig::is_random_walk_configured);
  RunConfig::fanout = std::vector<size_t>(fanout, fanout + num_fanout);
  RunConfig::is_khop_configured = true;
}

void samgraph_config_random_walk(size_t random_walk_length,
                                 double random_walk_restart_prob,
                                 size_t num_random_walk, size_t num_neighbor,
                                 size_t num_layer) {
  CHECK(!RunConfig::is_random_walk_configured &&
        !RunConfig::is_khop_configured);
  RunConfig::random_walk_length = random_walk_length;
  RunConfig::random_walk_restart_prob = random_walk_restart_prob;
  RunConfig::num_random_walk = num_random_walk;
  RunConfig::num_neighbor = num_neighbor;
  RunConfig::num_layer = num_layer;
  RunConfig::fanout = std::vector<size_t>(num_layer, num_neighbor);
  RunConfig::is_random_walk_configured = true;
}

void samgraph_init() {
  CHECK(RunConfig::is_configured);
  CHECK(RunConfig::is_khop_configured || RunConfig::is_random_walk_configured);
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

uint64_t samgraph_get_next_batch() {
  CHECK(Engine::Get()->IsInitialized() && !Engine::Get()->IsShutdown());

  // uint64_t key = Engine::Get()->GetBatchKey(epoch, step);
  auto graph = Engine::Get()->GetGraphPool()->GetGraphBatch();
  uint64_t key = graph->key;

  LOG(DEBUG) << "samgraph_get_next_batch encodeKey with key " << key;
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
  if (RunConfig::option_log_node_access_simple) {
    Profiler::Get().ReportNodeAccessSimple();
  }
  if (RunConfig::option_log_node_access) {
    Profiler::Get().ReportNodeAccess();
  }
}

void samgraph_data_init() {
  CHECK(RunConfig::is_configured);
  CHECK(RunConfig::is_khop_configured || RunConfig::is_random_walk_configured);
  Engine::Create();
  Engine::Get()->Init();

  LOG(INFO) << "SamGraph data has been initialized successfully";
}
void samgraph_sample_init(int device_type, int device_id) {
  CHECK(RunConfig::is_configured);
  CHECK(RunConfig::is_khop_configured || RunConfig::is_random_walk_configured);
  dist::DistEngine::Get()->SampleInit(device_type, device_id);

  LOG(INFO) << "SamGraph sample has been initialized successfully";
}
void samgraph_train_init(int device_type, int device_id) {
  CHECK(RunConfig::is_configured);
  CHECK(RunConfig::is_khop_configured || RunConfig::is_random_walk_configured);
  dist::DistEngine::Get()->TrainInit(device_type, device_id);

  LOG(INFO) << "SamGraph train has been initialized successfully";
}
void samgraph_extract_start(int count) {
  dist::DistEngine::Get()->StartExtract(count);
  LOG(INFO) << "SamGraph extract background thread start successfully";
}

void samgraph_trace_step_begin(uint64_t key, int item, uint64_t us) {
  Profiler::Get().TraceStepBegin(key, static_cast<TraceItem>(item), us);
}
void samgraph_trace_step_end(uint64_t key, int item, uint64_t us) {
  Profiler::Get().TraceStepEnd(key, static_cast<TraceItem>(item), us);
}
void samgraph_trace_step_begin_now(uint64_t key, int item) {
  Timer t;
  Profiler::Get().TraceStepBegin(key, static_cast<TraceItem>(item), t.TimePointMicro());
}
void samgraph_trace_step_end_now(uint64_t key, int item) {
  Timer t;
  Profiler::Get().TraceStepEnd(key, static_cast<TraceItem>(item), t.TimePointMicro());
}
void samgraph_dump_trace() {
  Profiler::Get().DumpTrace(std::cerr);
}
void samgraph_forward_barrier() {
  Engine::Get()->ForwardBarrier();
}

} // extern "c"

}  // namespace common
}  // namespace samgraph
