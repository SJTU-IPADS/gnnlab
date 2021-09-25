#include "profiler.h"

#include <cstdio>
#include <fstream>
#include <limits>
#include <numeric>
#include <unordered_map>

#ifdef __linux__
#include <parallel/algorithm>
#else
#include <algorithm>
#endif

#include "common.h"
#include "constant.h"
#include "engine.h"
#include "logging.h"
#include "run_config.h"
#include "cuda/pre_sampler.h"

namespace samgraph {
namespace common {

LogData::LogData(size_t num_logs) {
  vals.resize(num_logs, 0);
  bitmap.resize(num_logs, false);
  sum = 0;
  cnt = 0;
}

TraceEvent::TraceEvent() {
  begin = 0;
  end = 0;
}

TraceData::TraceData(size_t num_traces) {
  events.resize(num_traces, TraceEvent());
}

Profiler::Profiler() {
  size_t num_step_items = static_cast<size_t>(kNumLogStepItems);
  size_t num_step_logs = Engine::Get()->NumEpoch() * Engine::Get()->NumStep();
  size_t num_epoch_items = static_cast<size_t>(kNumLogEpochItems);
  size_t num_epoch_logs = Engine::Get()->NumEpoch();

  _init_data.resize(kNumLogInitItems, LogData(1));
  _step_data.resize(num_step_items, LogData(num_step_logs));
  _step_buf.resize(num_step_items);
  _epoch_data.resize(num_epoch_items, LogData(num_epoch_logs));
  _epoch_buf.resize(num_epoch_items);

  _step_trace.resize(kNumTraceItems, TraceData(num_step_logs));
  _num_step = Engine::Get()->NumStep();

  _node_access.resize(Engine::Get()->GetGraphDataset()->num_node, 0);
  _last_visit.resize(Engine::Get()->GetGraphDataset()->num_node, 0);
  _similarity.resize(num_step_logs);
}

void Profiler::ResetStepEpoch() {
  size_t num_step_items = static_cast<size_t>(kNumLogStepItems);
  size_t num_step_logs = Engine::Get()->NumEpoch() * Engine::Get()->NumStep();
  size_t num_epoch_items = static_cast<size_t>(kNumLogEpochItems);
  size_t num_epoch_logs = Engine::Get()->NumEpoch();
  _step_data.clear();
  _step_data.resize(num_step_items, LogData(num_step_logs));
  _step_buf.clear();
  _step_buf.resize(num_step_items);
  _epoch_data.clear();
  _epoch_data.resize(num_epoch_items, LogData(num_epoch_logs));
  _epoch_buf.clear();
  _epoch_buf.resize(num_epoch_items);

  _step_trace.clear();
  _step_trace.resize(kNumTraceItems, TraceData(num_step_logs));
  _num_step = Engine::Get()->NumStep();

  _node_access.clear();
  _node_access.resize(Engine::Get()->GetGraphDataset()->num_node, 0);
  _last_visit.clear();
  _last_visit.resize(Engine::Get()->GetGraphDataset()->num_node, 0);
  _similarity.clear();
  _similarity.resize(num_step_logs);
}

void Profiler::LogInit(LogInitItem item, double val) {
  uint64_t key = 0;
  _init_data[item].vals[key] = val;
  _init_data[item].sum += val;
  _init_data[item].cnt = _init_data[item].bitmap[key]
                         ? _init_data[item].cnt
                         : _init_data[item].cnt + 1;
  _init_data[item].bitmap[key] = true;
}
void Profiler::LogInitAdd(LogInitItem item, double val) {
  uint64_t key = 0;
  _init_data[item].vals[key] += val;
  _init_data[item].sum += val;
  _init_data[item].cnt = _init_data[item].bitmap[key]
                         ? _init_data[item].cnt
                         : _init_data[item].cnt + 1;
  _init_data[item].bitmap[key] = true;
}

void Profiler::LogStep(uint64_t key, LogStepItem item, double val) {
  size_t item_idx = static_cast<size_t>(item);
  _step_data[item_idx].vals[key] = val;
  _step_data[item_idx].sum += val;
  _step_data[item_idx].cnt = _step_data[item_idx].bitmap[key]
                                 ? _step_data[item_idx].cnt
                                 : _step_data[item_idx].cnt + 1;
  _step_data[item_idx].bitmap[key] = true;
}

void Profiler::LogStepAdd(uint64_t key, LogStepItem item, double val) {
  size_t item_idx = static_cast<size_t>(item);
  _step_data[item_idx].vals[key] += val;
  _step_data[item_idx].sum += val;
  _step_data[item_idx].cnt = _step_data[item_idx].bitmap[key]
                                 ? _step_data[item_idx].cnt
                                 : _step_data[item_idx].cnt + 1;
  _step_data[item_idx].bitmap[key] = true;
}

void Profiler::LogEpochAdd(uint64_t key, LogEpochItem item, double val) {
  uint64_t epoch = Engine::Get()->GetEpochFromKey(key);
  size_t item_idx = static_cast<size_t>(item);
  _epoch_data[item_idx].vals[epoch] += val;
  _epoch_data[item_idx].sum += val;
  _epoch_data[item_idx].cnt = _epoch_data[item_idx].bitmap[epoch]
                                  ? _epoch_data[item_idx].cnt
                                  : _epoch_data[item_idx].cnt + 1;
  _epoch_data[item_idx].bitmap[epoch] = true;
}

double Profiler::GetLogInitValue(LogStepItem item) {
  return _init_data[item].vals[0];
}

double Profiler::GetLogStepValue(uint64_t key, LogStepItem item) {
  size_t item_idx = static_cast<size_t>(item);
  return _step_data[item_idx].vals[key];
}

double Profiler::GetLogEpochValue(uint64_t epoch, LogEpochItem item) {
  size_t item_idx = static_cast<size_t>(item);
  return _epoch_data[item_idx].vals[epoch];
}

void Profiler::ReportInit() {
  std::string env_level = GetEnv(Constant::kEnvProfileLevel);

  int level = 0;
  if (env_level == "1") {
    level = 1;
  } else if (env_level == "2") {
    level = 2;
  } else if (env_level == "3") {
    level = 3;
  }
  if (level >= 1) {
    printf(
        "    [Init Profiler Level 1]\n"
        "        L1  load dataset %10.4lf | presample   %10.4lf | "
        "build cache %.4lf\n",
        _init_data[kLogInitL1LoadDataset].vals[0],
        _init_data[kLogInitL1Presample].vals[0],
        _init_data[kLogInitL1BuildCache].vals[0]);
  }
  if (level >= 2) {
    printf(
        "    [Init Profiler Level 2]\n"
        "        L2  presample: init   %10.4lf\n"
        "        L2  presample: sample %10.4lf | copy     %10.4lf\n"
        "        L2  presample: count  %10.4lf | sort     %10.4lf\n"
        "        L2  presample: reset  %10.4lf | get rank %10.4lf\n",
        _init_data[kLogInitL2PresampleInit].vals[0],
        _init_data[kLogInitL2PresampleSample].vals[0],
        _init_data[kLogInitL2PresampleCopy].vals[0],
        _init_data[kLogInitL2PresampleCount].vals[0],
        _init_data[kLogInitL2PresampleSort].vals[0],
        _init_data[kLogInitL2PresampleReset].vals[0],
        _init_data[kLogInitL2PresampleGetRank].vals[0]);
  }
}

void Profiler::ReportStep(uint64_t epoch, uint64_t step) {
  uint64_t key = Engine::Get()->GetBatchKey(epoch, step);

  size_t num_items = static_cast<size_t>(kNumLogStepItems);
  for (size_t i = 0; i < num_items; i++) {
    _step_buf[i] = _step_data[i].vals[key];
  }
  OutputStep(key, "Step");
}

void Profiler::ReportStepAverage(uint64_t epoch, uint64_t step) {
  uint64_t key = Engine::Get()->GetBatchKey(epoch, step);

  size_t num_items = static_cast<size_t>(kNumLogStepItems);
  for (size_t i = 0; i < num_items; i++) {
    double sum = _step_data[i].sum - _step_data[i].vals[0];
    size_t cnt = _step_data[i].cnt <= 1 ? 1 : _step_data[i].cnt - 1;
    _step_buf[i] = sum / cnt;
  }

  OutputStep(key, "Step(average)");
}

void Profiler::ReportEpoch(uint64_t epoch) {
  size_t num_items = static_cast<size_t>(kNumLogEpochItems);
  for (size_t i = 0; i < num_items; i++) {
    _epoch_buf[i] = _epoch_data[i].vals[epoch];
  }
  OutputEpoch(epoch, "Epoch");
}

void Profiler::ReportEpochAverage(uint64_t epoch) {
  size_t num_items = static_cast<size_t>(kNumLogEpochItems);
  for (size_t i = 0; i < num_items; i++) {
    double sum = _epoch_data[i].sum - _epoch_data[i].vals[0];
    size_t cnt = _epoch_data[i].cnt <= 1 ? 1 : _epoch_data[i].cnt - 1;
    _epoch_buf[i] = sum / cnt;
  }

  OutputEpoch(epoch, "Epoch(average)");
}

namespace {

#define F(name) #name ,
const char* trace_item_names[] = { TRACE_TYPES( F ) nullptr };
#undef F

std::string trace_item_to_string(TraceItem item) {
  return trace_item_names[(int)item];
}

struct TraceJsonHelper {
  std::string name;
  std::string ph;
  uint64_t pid, tid;
  uint64_t ts;
  std::string cat = "";
  uint64_t id = 0;
  TraceJsonHelper& set_name  (std::string n   ) {this->name = n    ; return *this;}
  TraceJsonHelper& set_begin (                ) {this->ph   = "B"  ; return *this;}
  TraceJsonHelper& set_end   (                ) {this->ph   = "E"  ; return *this;}
  TraceJsonHelper& set_pid   (uint64_t    pid ) {this->pid  = pid  ; return *this;}
  TraceJsonHelper& set_tid   (uint64_t    tid ) {this->tid  = tid  ; return *this;}
  TraceJsonHelper& set_ts    (uint64_t    ts  ) {this->ts   = ts   ; return *this;}
  TraceJsonHelper& set_cat   (std::string cat ) {this->cat  = cat  ; return *this;}
  TraceJsonHelper& set_id    (uint64_t    id  ) {this->id   = id   ; return *this;}
  std::string to_json(bool & first) {
    std::stringstream ss;
    if (! first) {
      ss << ",";
    } else {
      first = false;
    }
    ss  << "{"
        << "\"name\":" << "\"" << name << "\"" << ","
        << "\"ph\":"   << "\"" << ph << "\"" << ","
        << "\"pid\":"  << pid << ","
        << "\"tid\":"  << tid << ","
        << "\"ts\":"   << ts  << ","
        << "\"cat\":"  << "\"" << cat << "\"" << ","
        << "\"id\":"   << id
        << "}\n";
    
    return ss.str();
  }
};

}

void Profiler::DumpTrace(std::ostream &of) {
  if (RunConfig::option_dump_trace == false) return;
  bool first = true;
  of << "[\n";
  for (size_t item = 0; item < kNumTraceItems; item++) {
    uint64_t tid = 0;
    if (item < kL1Event_Sample) {
      tid = 0;
    } else if (item < kL1Event_Copy) {
      tid = 1;
    } else if (item < kL1Event_Convert) {
      tid = 2;
    } else {
      tid = 3;
    }
    for (size_t key = 0; key < _step_trace[item].events.size(); key++) {
      if (_step_trace[item].events[key].begin == 0) continue;
      auto & event = _step_trace[item].events[key];
      if (event.end == 0) {
        LOG(WARNING) << "An event without end";
        continue;
      }
      TraceJsonHelper tjs;
      tjs.set_name(trace_item_to_string((TraceItem)item) + "-" + std::to_string(key))
         .set_pid(0)
         .set_tid(tid);
      tjs.set_begin().set_ts(event.begin);
      of << tjs.to_json(first);
      tjs.set_end().set_ts(event.end);
      of << tjs.to_json(first);
    }
  }
  of << "]\n";
}

Profiler &Profiler::Get() {
  static Profiler inst;
  return inst;
}

void Profiler::OutputStep(uint64_t key, std::string type) {
  uint32_t epoch = Engine::Get()->GetEpochFromKey(key);
  uint32_t step = Engine::Get()->GetStepFromKey(key);

  std::string env_level = GetEnv(Constant::kEnvProfileLevel);

  int level = 0;
  if (env_level == "1") {
    level = 1;
  } else if (env_level == "2") {
    level = 2;
  } else if (env_level == "3") {
    level = 3;
  }

  if (level >= 1 && !RunConfig::UseGPUCache() && !RunConfig::UseDynamicGPUCache()) {
    printf(
        "    [%s Profiler Level 1 E%u S%u]\n"
        "        L1  sample         %10.4lf | copy         %10.4lf | "
        "convert time %.4lf | train  %.4lf\n"
        "        L1  feature nbytes %10s | label nbytes %10s\n"
        "        L1  id nbytes      %10s | graph nbytes %10s\n"
        "        L1  num nodes      %10lf | num samples  %10lf\n",
        type.c_str(), epoch, step, _step_buf[kLogL1SampleTime],
        _step_buf[kLogL1CopyTime], _step_buf[kLogL1ConvertTime],
        _step_buf[kLogL1TrainTime],
        ToReadableSize(_step_buf[kLogL1FeatureBytes]).c_str(),
        ToReadableSize(_step_buf[kLogL1LabelBytes]).c_str(),
        ToReadableSize(_step_buf[kLogL1IdBytes]).c_str(),
        ToReadableSize(_step_buf[kLogL1GraphBytes]).c_str(),
        _step_buf[kLogL1NumNode],_step_buf[kLogL1NumSample]);
  } else if (level >= 1 && RunConfig::UseDynamicGPUCache()) {
    printf(
        "    [%s Profiler Level 1 E%u S%u]\n"
        "        L1  sample         %10.4lf | send         %10.4lf\n"
        "        L1  recv           %10.4lf | copy         %10.4lf | "
        "convert time %.4lf | train  %.4lf\n"
        "        L1  feature nbytes %10s | label nbytes %10s\n"
        "        L1  id nbytes      %10s | graph nbytes %10s\n"
        "        L1  miss nbytes    %10s | hit rate %10s \n"
        "        L1  nodes          %10.1lf | cache rate %10s \n"
        "        L1  prefetch adv   %10.4lf | get nbr time %10.4lf\n",
        type.c_str(), epoch, step, _step_buf[kLogL1SampleTime],
        _step_buf[kLogL1SendTime], _step_buf[kLogL1RecvTime],
        _step_buf[kLogL1CopyTime], _step_buf[kLogL1ConvertTime],
        _step_buf[kLogL1TrainTime],
        ToReadableSize(_step_buf[kLogL1FeatureBytes]).c_str(),
        ToReadableSize(_step_buf[kLogL1LabelBytes]).c_str(),
        ToReadableSize(_step_buf[kLogL1IdBytes]).c_str(),
        ToReadableSize(_step_buf[kLogL1GraphBytes]).c_str(),
        ToReadableSize(_step_buf[kLogL1MissBytes]).c_str(),
        ToPercentage(1 - _step_buf[kLogL1MissBytes] / _step_buf[kLogL1FeatureBytes]).c_str(),
        _step_buf[kLogL1NumNode],
        ToPercentage(_step_buf[kLogL1NumNode] / Engine::Get()->GetGraphDataset()->num_node).c_str(),
        _step_buf[kLogL1PrefetchAdvanced], _step_buf[kLogL1GetNeighbourTime]);
  } else if (level >= 1) {
    printf(
        "    [%s Profiler Level 1 E%u S%u]\n"
        "        L1  sample         %10.4lf | send         %10.4lf\n"
        "        L1  recv           %10.4lf | copy         %10.4lf | "
        "convert time %.4lf | train  %.4lf\n"
        "        L1  feature nbytes %10s | label nbytes %10s\n"
        "        L1  id nbytes      %10s | graph nbytes %10s\n"
        "        L1  miss nbytes    %10s\n"
        "        L1  num nodes      %10lf | num samples  %10lf\n",
        type.c_str(), epoch, step, _step_buf[kLogL1SampleTime],
        _step_buf[kLogL1SendTime], _step_buf[kLogL1RecvTime],
        _step_buf[kLogL1CopyTime], _step_buf[kLogL1ConvertTime],
        _step_buf[kLogL1TrainTime],
        ToReadableSize(_step_buf[kLogL1FeatureBytes]).c_str(),
        ToReadableSize(_step_buf[kLogL1LabelBytes]).c_str(),
        ToReadableSize(_step_buf[kLogL1IdBytes]).c_str(),
        ToReadableSize(_step_buf[kLogL1GraphBytes]).c_str(),
        ToReadableSize(_step_buf[kLogL1MissBytes]).c_str(),
        _step_buf[kLogL1NumNode],_step_buf[kLogL1NumSample]);
  }

  if (level >= 2 && !RunConfig::UseGPUCache()) {
    printf(
        "    [%s Profiler Level 2 E%u S%u]\n"
        "        L2  shuffle     %.4lf | core sample  %.4lf | id remap  %.4lf\n"
        "        L2  graph copy  %.4lf | id copy      %.4lf | extract   %.4lf |"
        " feat copy %.4lf\n"
        "        L2  last layer sample time %.4lf | size %.4lf\n",
        type.c_str(), epoch, step, _step_buf[kLogL2ShuffleTime],
        _step_buf[kLogL2CoreSampleTime], _step_buf[kLogL2IdRemapTime],
        _step_buf[kLogL2GraphCopyTime], _step_buf[kLogL2IdCopyTime],
        _step_buf[kLogL2ExtractTime], _step_buf[kLogL2FeatCopyTime],
        _step_buf[kLogL2LastLayerTime], _step_buf[kLogL2LastLayerSize]);
  } else if (level >= 2) {
    printf(
        "    [%s Profiler Level 2 E%u S%u]\n"
        "        L2  shuffle     %.4lf | core sample  %.4lf | "
        "id remap        %.4lf\n"
        "        L2  graph copy  %.4lf | id copy      %.4lf | "
        "cache feat copy %.4lf\n"
        "        L2  last layer sample time %.4lf | size %.4lf\n",
        type.c_str(), epoch, step, _step_buf[kLogL2ShuffleTime],
        _step_buf[kLogL2CoreSampleTime], _step_buf[kLogL2IdRemapTime],
        _step_buf[kLogL2GraphCopyTime], _step_buf[kLogL2IdCopyTime],
        _step_buf[kLogL2CacheCopyTime],
        _step_buf[kLogL2LastLayerTime], _step_buf[kLogL2LastLayerSize]);
  }

  if (level >= 3 && !RunConfig::UseGPUCache()) {
    printf(
        "     [%s Profiler Level 3 E%u S%u]\n"
        "        L3  khop sample coo  %.4lf | khop sort coo     %.4lf | "
        "khop count edge   %.4lf | khop compact edge %.4lf\n"
        "        L3  walk sample coo  %.4lf | walk topk total   %.4lf | "
        "walk topk step1   %.4lf | walk topk step2   %.4lf\n"
        "        L3  walk topk step3  %.4lf | walk topk step4   %.4lf | "
        "walk topk step5   %.4lf\n"
        "        L3  walk topk step6  %.4lf | walk topk step7   %.4lf | "
        "walk topk step8   %.4lf\n"
        "        L3  walk topk step9  %.4lf | walk topk step10  %.4lf | "
        "walk topk step11  %.4lf\n"
        "        L3  remap unique     %.4lf | remap populate    %.4lf | "
        "remap mapnode     %.4lf | remap mapedge     %.4lf\n",
        type.c_str(), epoch, step, _step_buf[kLogL3KHopSampleCooTime],
        _step_buf[kLogL3KHopSampleSortCooTime],
        _step_buf[kLogL3KHopSampleCountEdgeTime],
        _step_buf[kLogL3KHopSampleCompactEdgesTime],
        _step_buf[kLogL3RandomWalkSampleCooTime],
        _step_buf[kLogL3RandomWalkTopKTime],
        _step_buf[kLogL3RandomWalkTopKStep1Time],
        _step_buf[kLogL3RandomWalkTopKStep2Time],
        _step_buf[kLogL3RandomWalkTopKStep3Time],
        _step_buf[kLogL3RandomWalkTopKStep4Time],
        _step_buf[kLogL3RandomWalkTopKStep5Time],
        _step_buf[kLogL3RandomWalkTopKStep6Time],
        _step_buf[kLogL3RandomWalkTopKStep7Time],
        _step_buf[kLogL3RandomWalkTopKStep8Time],
        _step_buf[kLogL3RandomWalkTopKStep9Time],
        _step_buf[kLogL3RandomWalkTopKStep10Time],
        _step_buf[kLogL3RandomWalkTopKStep11Time],
        _step_buf[kLogL3RemapFillUniqueTime],
        _step_buf[kLogL3RemapPopulateTime], _step_buf[kLogL3RemapMapNodeTime],
        _step_buf[kLogL3RemapMapEdgeTime]);
  } else if (level >= 3) {
    printf(
        "    [%s Profiler Level 3 E%u S%u]\n"
        "        L3  khop sample coo  %.4lf | khop sort coo      %.4lf | "
        "khop count edge     %.4lf | khop compact edge %.4lf\n"
        "        L3  walk sample coo  %.4lf | walk topk total    %.4lf | "
        "walk topk step1     %.4lf | walk topk step2   %.4lf\n"
        "        L3  walk topk step3  %.4lf | walk topk step4    %.4lf | "
        "walk topk step5     %.4lf\n"
        "        L3  walk topk step6  %.4lf | walk topk step7    %.4lf | "
        "walk topk step8     %.4lf\n"
        "        L3  walk topk step9  %.4lf | walk topk step10   %.4lf | "
        "walk topk step11    %.4lf\n"
        "        L3  remap unique     %.4lf | remap populate     %.4lf | "
        "remap mapnode       %.4lf | remap mapedge     %.4lf\n"
        "        L3  cache get_index  %.4lf | cache copy_index   %.4lf | "
        "cache extract_miss  %.4lf\n"
        "        L3  cache copy_miss  %.4lf | cache combine_miss %.4lf | "
        "cache combine cache %.4lf\n",
        type.c_str(), epoch, step, _step_buf[kLogL3KHopSampleCooTime],
        _step_buf[kLogL3KHopSampleSortCooTime],
        _step_buf[kLogL3KHopSampleCountEdgeTime],
        _step_buf[kLogL3KHopSampleCompactEdgesTime],
        _step_buf[kLogL3RandomWalkSampleCooTime],
        _step_buf[kLogL3RandomWalkTopKTime],
        _step_buf[kLogL3RandomWalkTopKStep1Time],
        _step_buf[kLogL3RandomWalkTopKStep2Time],
        _step_buf[kLogL3RandomWalkTopKStep3Time],
        _step_buf[kLogL3RandomWalkTopKStep4Time],
        _step_buf[kLogL3RandomWalkTopKStep5Time],
        _step_buf[kLogL3RandomWalkTopKStep6Time],
        _step_buf[kLogL3RandomWalkTopKStep7Time],
        _step_buf[kLogL3RandomWalkTopKStep8Time],
        _step_buf[kLogL3RandomWalkTopKStep9Time],
        _step_buf[kLogL3RandomWalkTopKStep10Time],
        _step_buf[kLogL3RandomWalkTopKStep11Time],
        _step_buf[kLogL3RemapFillUniqueTime],
        _step_buf[kLogL3RemapPopulateTime], _step_buf[kLogL3RemapMapNodeTime],
        _step_buf[kLogL3RemapMapEdgeTime], _step_buf[kLogL3CacheGetIndexTime],
        _step_buf[KLogL3CacheCopyIndexTime],
        _step_buf[kLogL3CacheExtractMissTime],
        _step_buf[kLogL3CacheCopyMissTime],
        _step_buf[kLogL3CacheCombineMissTime],
        _step_buf[kLogL3CacheCombineCacheTime]);
  }
}

void Profiler::OutputEpoch(uint64_t epoch, std::string type) {
  printf(
      "  [%s Profiler E%u]\n"
      "      total %.4lf | sample %.4lf | copy %.4lf | train %.4lf\n",
      type.c_str(), static_cast<uint32_t>(epoch),
      _epoch_buf[kLogEpochTotalTime], _epoch_buf[kLogEpochSampleTime],
      _epoch_buf[kLogEpochCopyTime], _epoch_buf[kLogEpochTrainTime]);
}

void Profiler::LogNodeAccess(uint64_t key, const IdType *input,
                             size_t num_input) {
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for (size_t i = 0; i < num_input; ++i) {
    _node_access[input[i]]++;
  }

  size_t similarity_count = 0;
#pragma omp parallel for num_threads(RunConfig::omp_thread_num) reduction(+ : similarity_count)
  for (size_t i = 0; i < num_input; ++i) {
    if (_last_visit[input[i]]) {
      similarity_count++;
    }
  }

#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for (size_t i = 0; i < _last_visit.size(); ++i) {
    _last_visit[i] = 0;
  }

#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for (size_t i = 0; i < num_input; ++i) {
    _last_visit[input[i]] = 1;
  }

  _similarity[key] = similarity_count;
}

void Profiler::ReportNodeAccess() {
  LOG(INFO) << "Writing the node access data to file...";

  double num_nodes =
      static_cast<double>(Engine::Get()->GetGraphDataset()->num_node);

  const IdType *in_degrees = static_cast<const IdType *>(
      Engine::Get()->GetGraphDataset()->in_degrees->Data());
  const IdType *out_degrees = static_cast<const IdType *>(
      Engine::Get()->GetGraphDataset()->out_degrees->Data());
  std::ofstream ofs0(Constant::kNodeAccessLogFile + GetTimeString() +
                         Constant::kNodeAccessFileSuffix,
                     std::ofstream::out | std::ofstream::trunc);
  std::ofstream ofs1(Constant::kNodeAccessFrequencyFile + GetTimeString() +
                         Constant::kNodeAccessFileSuffix,
                     std::ofstream::out | std::ofstream::trunc);
  std::ofstream ofs2(Constant::kNodeAccessSimilarityFile + GetTimeString() +
                         Constant::kNodeAccessFileSuffix,
                     std::ofstream::out | std::ofstream::trunc);

  // (frequency, nodeid)
  std::vector<std::pair<size_t, IdType>> records;
  // (frequency, count): how many nodes are accessed 'frequency' time
  std::vector<std::pair<size_t, size_t>> frequency;
  // (frequency, count): how many nodes are accessed 'frequency' time
  std::unordered_map<size_t, size_t> frequency_map;
  // (frequency, sum indegree)
  std::unordered_map<size_t, size_t> sum_indegree_map;
  // (frequency, min indegree)
  std::unordered_map<size_t, IdType> min_indegree_map;
  // (frequency, max indegree)
  std::unordered_map<size_t, IdType> max_indegree_map;
  // (frequency, sum outdegree)
  std::unordered_map<size_t, size_t> sum_outdegree_map;
  // (frequency, min outdegree)
  std::unordered_map<size_t, IdType> min_outdegree_map;
  // (frequency, max indegree)
  std::unordered_map<size_t, IdType> max_outdegree_map;
  // how many nodes are accessed
  double count_sum = 0;
  // how many times are nodes accessed
  double access_sum = 0;
  // count's prefix sum
  double count_percentypee_prefix_sum = 0;
  // access's prefix sum
  double access_percentypee_prefix_sum = 0;

  for (IdType nodeid = 0; nodeid < _node_access.size(); nodeid++) {
    if (_node_access[nodeid] > 0) {
      size_t frequency = _node_access[nodeid];
      count_sum++;
      records.push_back({frequency, nodeid});
      frequency_map[frequency]++;
      access_sum += frequency;

      if (min_indegree_map[frequency] == 0) {
        min_indegree_map[frequency] = std::numeric_limits<IdType>::max();
      }
      if (min_outdegree_map[frequency] == 0) {
        min_outdegree_map[frequency] = std::numeric_limits<IdType>::max();
      }

      sum_indegree_map[frequency] += in_degrees[nodeid];
      min_indegree_map[frequency] =
          Min(min_indegree_map[frequency], in_degrees[nodeid]);
      max_indegree_map[frequency] =
          Max(max_indegree_map[frequency], in_degrees[nodeid]);
      sum_outdegree_map[frequency] += out_degrees[nodeid];
      min_outdegree_map[frequency] =
          Min(min_outdegree_map[frequency], out_degrees[nodeid]);
      max_outdegree_map[frequency] =
          Min(min_outdegree_map[frequency], out_degrees[nodeid]);
    }
  }

  for (auto &p : frequency_map) {
    frequency.push_back({p.first, p.second});
  }

  // Sorted by frequency
#ifdef __linux__
  __gnu_parallel::sort(records.begin(), records.end(),
                       std::greater<std::pair<size_t, IdType>>());
  __gnu_parallel::sort(frequency.begin(), frequency.end(),
                       std::greater<std::pair<size_t, size_t>>());
#else
  std::sort(records.begin(), records.end(),
            std::greater<std::pair<size_t, IdType>>());
  std::sort(frequency.begin(), frequency.end(),
            std::greater<std::pair<size_t, size_t>>());
#endif

  for (auto &p : records) {
    IdType nodeid = p.second;
    size_t access = p.first;
    ofs0 << nodeid << " " << access << " " << in_degrees[nodeid] << " "
         << out_degrees[nodeid] << "\n";
  }

  for (auto &p : frequency) {
    size_t frequency = p.first;
    size_t count = p.second;
    double count_percentypee = static_cast<double>(count) / num_nodes;
    count_percentypee_prefix_sum += count_percentypee;

    size_t access = frequency * count;
    double access_percentypee = static_cast<double>(access) / access_sum;
    access_percentypee_prefix_sum += access_percentypee;

    double average_indegree = static_cast<double>(sum_indegree_map[frequency]) /
                              static_cast<double>(count);
    double average_outdegree =
        static_cast<double>(sum_outdegree_map[frequency]) /
        static_cast<double>(count);

    ofs1 << frequency << " " << count << " " << count_percentypee << " "
         << count_percentypee_prefix_sum << " " << access << " "
         << access_percentypee << " " << access_percentypee_prefix_sum << " "
         << min_indegree_map[frequency] << " " << average_indegree << " "
         << max_indegree_map[frequency] << " " << min_outdegree_map[frequency]
         << " " << average_outdegree << " " << max_outdegree_map[frequency]
         << "\n";
  }

  for (size_t i = 0; i < _similarity.size(); i++) {
    double similarity_percentypee =
        _similarity[i] / _step_data[kLogL1NumNode].vals[i];
    ofs2 << i << " " << _step_data[kLogL1NumNode].vals[i] << " "
         << _similarity[i] << " " << similarity_percentypee << "\n";
  }

  ofs0.close();
  ofs1.close();
  ofs2.close();
}

void Profiler::ReportNodeAccessSimple() {
  LOG(INFO) << "Writing the node access data to file...";

  // std::ofstream ofs0(Constant::kNodeAccessLogFile + GetTimeString() +
  //                        Constant::kNodeAccessFileSuffix,
  //                    std::ofstream::out | std::ofstream::trunc);
  std::ofstream ofs1(Constant::kNodeAccessOptimalCacheBinFile + GetTimeString() +
                         Constant::kNodeAccessFileSuffix,
                     std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
  std::ofstream ofs2(Constant::kNodeAccessOptimalCacheHitFile + GetTimeString() +
                         Constant::kNodeAccessFileSuffix,
                     std::ofstream::out | std::ofstream::trunc);
  size_t num_nodes = _node_access.size();
  // (frequency, nodeid)
  std::vector<std::pair<size_t, IdType>> records(num_nodes, {0, 0});
  // how many times are nodes accessed
  size_t frequency_sum = 0;

#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for (IdType nodeid = 0; nodeid < _node_access.size(); nodeid++) {
    size_t frequency = _node_access[nodeid];
    records[nodeid] = {frequency, nodeid};
  }

  // Sorted by frequency
#ifdef __linux__
  __gnu_parallel::sort(records.begin(), records.end(),
                       std::greater<std::pair<size_t, IdType>>());
#else
  std::sort(records.begin(), records.end(),
            std::greater<std::pair<size_t, IdType>>());
#endif

  for (auto & p : records) {
    size_t frequency = p.first;
    IdType nodeid = p.second;
    ofs1.write(reinterpret_cast<char*>(&nodeid), sizeof(IdType));
    frequency_sum += frequency;
    p.first = frequency_sum;
  }

  for (int cache_rate = 0; cache_rate <= 100; cache_rate++) {
    double hit_rate = records[static_cast<int>((static_cast<double>(cache_rate) / 100) * num_nodes)].first / static_cast<double>(frequency_sum);
    ofs2 << cache_rate << "\t" << hit_rate << "\n";
  }

  for (auto &p : records) {
    IdType nodeid = p.second;
    size_t prefix = p.first;
    // ofs0 << nodeid << " " << _node_access[nodeid] << " " << prefix << " "
    //      << static_cast<double>(prefix) / frequency_sum << "\n";
  }

  // ofs0.close();
  ofs1.close();
  ofs2.close();
}


void Profiler::ReportPreSampleSimilarity() {
  auto cache_node_tensor = cuda::PreSampler::Get()->GetRankNode();
  auto pre_sample_freq_tensor = cuda::PreSampler::Get()->GetFreq();
  const IdType* cache_nodes = static_cast<const IdType*>(cache_node_tensor->Data());
  const IdType* pre_sample_freq = static_cast<const IdType*>(pre_sample_freq_tensor->Data());
  std::ofstream ofs0(Constant::kNodeAccessPreSampleSimFile + GetTimeString() +
                         Constant::kNodeAccessFileSuffix,
                     std::ofstream::out | std::ofstream::trunc);
  for (IdType rank = 0; rank < Engine::Get()->GetGraphDataset()->num_node; rank++) {
    ofs0 << cache_nodes[rank] << " " << pre_sample_freq[rank] << " " << _node_access[cache_nodes[rank]] << "\n";
  }
  ofs0.close();
}

}  // namespace common
}  // namespace samgraph
