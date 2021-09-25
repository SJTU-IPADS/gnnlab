#include "operation.h"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "./dist/dist_engine.h"
#include "common.h"
#include "constant.h"
#include "engine.h"
#include "logging.h"
#include "profiler.h"
#include "run_config.h"
#include "timer.h"

namespace samgraph {
namespace common {

extern "C" {

void samgraph_config(const char **config_keys, const char **config_values,
                     const size_t num_config_items) {
  using RC = RunConfig;
  CHECK(!RC::is_configured);

  std::unordered_map<std::string, std::string> configs;

  for (size_t i = 0; i < num_config_items; i++) {
    std::string k(config_keys[i]);
    std::string v(config_values[i]);
    configs[k] = v;
  }

  CHECK(configs.count("dataset_path"));
  CHECK(configs.count("_arch"));
  CHECK(configs.count("_sample_type"));
  CHECK(configs.count("batch_size"));
  CHECK(configs.count("num_epoch"));
  CHECK(configs.count("_cache_policy"));
  CHECK(configs.count("cache_percentage"));
  CHECK(configs.count("max_sampling_jobs"));
  CHECK(configs.count("max_copying_jobs"));
  CHECK(configs.count("omp_thread_num"));

  RC::raw_configs = configs;
  RC::dataset_path = configs["dataset_path"];
  RC::run_arch = static_cast<RunArch>(std::stoi(configs["_arch"]));
  RC::sample_type = static_cast<SampleType>(std::stoi(configs["_sample_type"]));
  RC::batch_size = std::stoull(configs["batch_size"]);
  RC::num_epoch = std::stoull(configs["num_epoch"]);
  RC::cache_policy =
      static_cast<CachePolicy>(std::stoi(configs["_cache_policy"]));
  RC::cache_percentage = std::stod(configs["cache_percentage"]);

  RC::max_sampling_jobs = std::stoull(configs["max_sampling_jobs"]);
  RC::max_copying_jobs = std::stoull(configs["max_copying_jobs"]);
  RC::omp_thread_num = std::stoi(configs["omp_thread_num"]);

  if (RC::run_arch != kArch5) {
    CHECK(configs.count("sampler_ctx"));
    CHECK(configs.count("trainer_ctx"));
    RC::sampler_ctx = Context(configs["sampler_ctx"]);
    RC::trainer_ctx = Context(configs["trainer_ctx"]);
  } else {
    CHECK(configs.count("num_sample_worker"));
    CHECK(configs.count("num_train_worker"));
    RC::num_sample_worker = std::stoull(configs["num_sample_worker"]);
    RC::num_train_worker = std::stoull(configs["num_train_worker"]);
  }

  if (RC::sample_type != kRandomWalk) {
    // configure khop
    CHECK(configs.count("num_fanout"));
    CHECK(configs.count("fanout"));

    size_t num_fanout = std::stoull(configs["num_fanout"]);
    std::stringstream ss(configs["fanout"]);
    for (size_t i = 0; i < num_fanout; i++) {
      size_t fanout;
      ss >> fanout;
      RC::fanout.push_back(fanout);
    }
  } else {
    // configure random walk
    CHECK(configs.count("random_walk_length"));
    CHECK(configs.count("random_walk_restart_prob"));
    CHECK(configs.count("num_random_walk"));
    CHECK(configs.count("num_neighbor"));
    CHECK(configs.count("num_layer"));

    RC::random_walk_length = std::stoull(configs["random_walk_length"]);
    RC::random_walk_restart_prob =
        std::stod(configs["random_walk_restart_prob"]);
    RC::num_random_walk = std::stoull(configs["num_random_walk"]);
    RC::num_neighbor = std::stoull(configs["num_neighbor"]);
    RC::num_layer = std::stoull(configs["num_layer"]);
    RC::fanout = std::vector<size_t>(RC::num_layer, RC::num_neighbor);
  }

  if (configs.count("barriered_epoch") > 0) {
    RunConfig::barriered_epoch = std::stoi(configs["barriered_epoch"]);
    LOG(DEBUG) << "barriered_epoch=" << RunConfig::barriered_epoch;
  } else {
    RunConfig::barriered_epoch = 0;
  }

  if (configs.count("presample_epoch") > 0) {
    RunConfig::presample_epoch = std::stoi(configs["presample_epoch"]);
    LOG(DEBUG) << "presample_epoch=" << RunConfig::presample_epoch;
  } else {
    RunConfig::presample_epoch = 0;
  }

  RC::LoadConfigFromEnv();
  LOG(INFO) << "Use " << RunConfig::sample_type << " sampling algorithm";
  RC::is_configured = true;
}

void samgraph_init() {
  CHECK(RunConfig::is_configured);
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

void samgraph_report_init() { Profiler::Get().ReportInit(); }

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

void samgraph_trace_step_begin(uint64_t key, int item, uint64_t us) {
  Profiler::Get().TraceStepBegin(key, static_cast<TraceItem>(item), us);
}

void samgraph_trace_step_end(uint64_t key, int item, uint64_t us) {
  Profiler::Get().TraceStepEnd(key, static_cast<TraceItem>(item), us);
}

void samgraph_trace_step_begin_now(uint64_t key, int item) {
  Timer t;
  Profiler::Get().TraceStepBegin(key, static_cast<TraceItem>(item),
                                 t.TimePointMicro());
}

void samgraph_trace_step_end_now(uint64_t key, int item) {
  Timer t;
  Profiler::Get().TraceStepEnd(key, static_cast<TraceItem>(item),
                               t.TimePointMicro());
}

void samgraph_dump_trace() { Profiler::Get().DumpTrace(std::cerr); }

void samgraph_forward_barrier() { Engine::Get()->ForwardBarrier(); }

void samgraph_data_init() {
  CHECK(RunConfig::is_configured);
  Engine::Create();
  Engine::Get()->Init();

  LOG(INFO) << "SamGraph data has been initialized successfully";
}

void samgraph_sample_init(int worker_id, const char*ctx) {
  CHECK(RunConfig::is_configured);
  dist::DistEngine::Get()->SampleInit(worker_id, Context(std::string(ctx)));

  LOG(INFO) << "SamGraph sample has been initialized successfully";
}

void samgraph_train_init(int worker_id, const char*ctx) {
  CHECK(RunConfig::is_configured);
  dist::DistEngine::Get()->TrainInit(worker_id, Context(std::string(ctx)));

  LOG(INFO) << "SamGraph train has been initialized successfully";
}

void samgraph_extract_start(int count) {
  dist::DistEngine::Get()->StartExtract(count);
  LOG(INFO) << "SamGraph extract background thread start successfully";
}

}  // extern "c"

}  // namespace common
}  // namespace samgraph
