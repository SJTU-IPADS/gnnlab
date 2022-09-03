/*
 * Copyright 2022 Institute of Parallel and Distributed Systems, Shanghai Jiao Tong University
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

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
  uint64_t ts, dur;
  std::string cat = "";
  uint64_t id = 0;
  TraceJsonHelper& set_name  (std::string n   ) {this->name = n    ; return *this;}
  TraceJsonHelper& set_exe   (                ) {this->ph   = "X"  ; return *this;}
  TraceJsonHelper& set_pid   (uint64_t    pid ) {this->pid  = pid  ; return *this;}
  TraceJsonHelper& set_tid   (uint64_t    tid ) {this->tid  = tid  ; return *this;}
  TraceJsonHelper& set_ts    (uint64_t    ts  ) {this->ts   = ts   ; return *this;}
  TraceJsonHelper& set_dur   (uint64_t    dur ) {this->dur  = dur  ; return *this;}
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
        << "\"ts\":"   << ts  << ",";
    if (this->ph == "X") {
      ss << "\"dur\":" << dur << ",";
    }
    ss  << "\"cat\":"  << "\"" << cat << "\"" << ","
        << "\"id\":"   << id
        << "}\n";

    return ss.str();
  }
};

}

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

  if (RunConfig::option_log_node_access || RunConfig::option_log_node_access_simple) {
    _node_access.resize(Engine::Get()->GetGraphDataset()->num_node, 0);
    _last_visit.resize(Engine::Get()->GetGraphDataset()->num_node, 0);
    _similarity.resize(num_step_logs);
    _epoch_last_visit.resize(Engine::Get()->GetGraphDataset()->num_node, 0);
    _epoch_cur_visit.resize(Engine::Get()->GetGraphDataset()->num_node, 0);
    _epoch_similarity.resize(num_epoch_logs);
  }
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

  if (RunConfig::option_log_node_access || RunConfig::option_log_node_access_simple) {
    _node_access.clear();
    _node_access.resize(Engine::Get()->GetGraphDataset()->num_node, 0);
    _last_visit.clear();
    _last_visit.resize(Engine::Get()->GetGraphDataset()->num_node, 0);
    _similarity.clear();
    _similarity.resize(num_step_logs);
    _epoch_last_visit.clear();
    _epoch_last_visit.resize(Engine::Get()->GetGraphDataset()->num_node, 0);
    _epoch_cur_visit.clear();
    _epoch_cur_visit.resize(Engine::Get()->GetGraphDataset()->num_node, 0);
    _epoch_similarity.clear();
    _epoch_similarity.resize(num_epoch_logs);
  }
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

double Profiler::GetLogInitValue(LogInitItem item) {
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
        "        L1  init %10.4lf | sampler init %10.4lf | "
        "trainer init %.4lf\n",
        _init_data[kLogInitL1Common].vals[0],
        _init_data[kLogInitL1Sampler].vals[0],
        _init_data[kLogInitL1Trainer].vals[0]);
  }
  if (level >= 2) {
    printf(
        "    [Init Profiler Level 2]\n"
        "        L2  load ds     %10.4lf | init queue       %10.4lf\n"
        "        L2  presample   %10.4lf | build interal ds %10.4lf\n"
        "        L2  build cache %10.4lf\n",
        _init_data[kLogInitL2LoadDataset].vals[0],
        _init_data[kLogInitL2DistQueue].vals[0],
        _init_data[kLogInitL2Presample].vals[0],
        _init_data[kLogInitL2InternalState].vals[0],
        _init_data[kLogInitL2BuildCache].vals[0]);
  }
  if (level >= 3) {
    printf(
        "    [Init Profiler Level 3]\n"
        "        L3  load dataset: mmap %10.4lf | copy     %10.4lf\n"
        "        L3  dist queue: alloc  %10.4lf | pin      %10.4lf | "
                                "push %10.4lf\n"
        "        L3  presample: init    %10.4lf\n"
        "        L3  presample: sample  %10.4lf | copy     %10.4lf\n"
        "        L3  presample: count   %10.4lf | sort     %10.4lf\n"
        "        L3  presample: reset   %10.4lf | get rank %10.4lf\n"
        "        L3  internal: cuda ctx %10.4lf | cuda stream %10.4lf\n",
        _init_data[kLogInitL3LoadDatasetMMap].vals[0],
        _init_data[kLogInitL3LoadDatasetCopy].vals[0],
        _init_data[kLogInitL3DistQueueAlloc].vals[0],
        _init_data[kLogInitL3DistQueuePin].vals[0],
        _init_data[kLogInitL3DistQueuePush].vals[0],
        _init_data[kLogInitL3PresampleInit].vals[0],
        _init_data[kLogInitL3PresampleSample].vals[0],
        _init_data[kLogInitL3PresampleCopy].vals[0],
        _init_data[kLogInitL3PresampleCount].vals[0],
        _init_data[kLogInitL3PresampleSort].vals[0],
        _init_data[kLogInitL3PresampleReset].vals[0],
        _init_data[kLogInitL3PresampleGetRank].vals[0],
        _init_data[kLogInitL3InternalStateCreateCtx].vals[0],
        _init_data[kLogInitL3InternalStateCreateStream].vals[0]);
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
    if (_step_data[i].cnt <= 1) {
      _step_buf[i] = 0;
      continue;
    }
    // skip first epoch
    double sum = _step_data[i].sum;
    size_t cnt = _step_data[i].cnt;
    for (size_t current_key = 0; Engine::Get()->GetEpochFromKey(current_key) == 0; current_key++) {
      if (_step_data[i].bitmap[current_key] == false) continue;
      sum -= _step_data[i].vals[current_key];
      cnt --;
    }
    if (cnt == 0) {
      CHECK_LE(std::abs(sum), 1e-8) << " sum is " << sum;
      cnt = 1;
    }
    _step_buf[i] = sum / cnt;
  }

  OutputStep(key, "Step(average)");
}

template<typename ReduceOp>
void Profiler::PrepareStepReduce(uint64_t epoch, uint64_t step, const double init, ReduceOp op) {
  uint64_t key = Engine::Get()->GetBatchKey(epoch, step);

  size_t num_items = static_cast<size_t>(kNumLogStepItems);
  for (size_t i = 0; i < num_items; i++) {
    if (_step_data[i].cnt <= 1) {
      _step_buf[i] = 0;
      continue;
    }
    double reduce_val = init;
    for (size_t current_key = 0; current_key < key; current_key++) {
      // skip first epoch
      if (Engine::Get()->GetEpochFromKey(current_key) == 0) continue;
      if (_step_data[i].bitmap[current_key] == false) continue;
      reduce_val = op(reduce_val, _step_data[i].vals[current_key]);
    }
    _step_buf[i] = reduce_val;
  }
}

void Profiler::ReportStepMax(uint64_t epoch, uint64_t step) {
  uint64_t key = Engine::Get()->GetBatchKey(epoch, step);
  auto reduce_op = [](const double & a, const double & b){ return std::max(a, b); };
  PrepareStepReduce<>(epoch, step, 0, reduce_op);
  OutputStep(key, "Step(max)");
}
void Profiler::ReportStepMin(uint64_t epoch, uint64_t step) {
  uint64_t key = Engine::Get()->GetBatchKey(epoch, step);
  auto reduce_op = [](const double & a, const double & b){ return std::min(a, b); };
  PrepareStepReduce<>(epoch, step, std::numeric_limits<double>::max(), reduce_op);
  OutputStep(key, "Step(min)");
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
      tjs.set_exe().set_ts(event.begin).set_dur(event.end - event.begin);
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

  /*if (level >= 1 && !RunConfig::UseGPUCache() && !RunConfig::UseDynamicGPUCache()) {
    printf(
        "    [%s Profiler Level 1 E%u S%u]\n"
        "        L1  sample         %10.6lf | send         %10.6lf\n"
        "        L1  recv           %10.6lf | copy         %10.6lf | "
        "convert time %.6lf | train  %.6lf\n"
        "        L1  feature nbytes %10s | label nbytes %10s\n"
        "        L1  id nbytes      %10s | graph nbytes %10s\n"
        "        L1  num nodes      %10.0lf | num samples  %10.0lf\n",
        type.c_str(), epoch, step, _step_buf[kLogL1SampleTime],
        _step_buf[kLogL1SendTime], _step_buf[kLogL1RecvTime],
        _step_buf[kLogL1CopyTime], _step_buf[kLogL1ConvertTime],
        _step_buf[kLogL1TrainTime],
        ToReadableSize(_step_buf[kLogL1FeatureBytes]).c_str(),
        ToReadableSize(_step_buf[kLogL1LabelBytes]).c_str(),
        ToReadableSize(_step_buf[kLogL1IdBytes]).c_str(),
        ToReadableSize(_step_buf[kLogL1GraphBytes]).c_str(),
        _step_buf[kLogL1NumNode],_step_buf[kLogL1NumSample]);
  } else */ if (level >= 1 && RunConfig::UseDynamicGPUCache()) {
    printf(
        "    [%s Profiler Level 1 E%u S%u]\n"
        "        L1  sample         %10.6lf | send         %10.6lf\n"
        "        L1  recv           %10.6lf | copy         %10.6lf | "
        "convert time %.6lf | train  %.6lf\n"
        "        L1  feature nbytes %10s | label nbytes %10s\n"
        "        L1  id nbytes      %10s | graph nbytes %10s\n"
        "        L1  miss nbytes    %10s | hit rate %10s \n"
        "        L1  nodes          %10.0lf | cache rate %10s \n"
        "        L1  prefetch adv   %10.6lf | get nbr time %10.6lf\n",
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
        "        L1  sample         %10.6lf | send         %10.6lf\n"
        "        L1  recv           %10.6lf | copy         %10.6lf | "
        "convert time %.6lf | train  %.6lf\n"
        "        L1  feature nbytes %10s | label nbytes %10s\n"
        "        L1  id nbytes      %10s | graph nbytes %10s\n"
        "        L1  miss nbytes    %10s | remote nbytes %10s\n"
        "        L1  num nodes      %10.0lf | num samples  %10.0lf\n",
        type.c_str(), epoch, step, _step_buf[kLogL1SampleTime],
        _step_buf[kLogL1SendTime], _step_buf[kLogL1RecvTime],
        _step_buf[kLogL1CopyTime], _step_buf[kLogL1ConvertTime],
        _step_buf[kLogL1TrainTime],
        ToReadableSize(_step_buf[kLogL1FeatureBytes]).c_str(),
        ToReadableSize(_step_buf[kLogL1LabelBytes]).c_str(),
        ToReadableSize(_step_buf[kLogL1IdBytes]).c_str(),
        ToReadableSize(_step_buf[kLogL1GraphBytes]).c_str(),
        ToReadableSize(_step_buf[kLogL1MissBytes]).c_str(),
        ToReadableSize(_step_buf[kLogL1RemoteBytes]).c_str(),
        _step_buf[kLogL1NumNode],_step_buf[kLogL1NumSample]);
  }

  /*if (level >= 2 && !RunConfig::UseGPUCache()) {
    printf(
        "    [%s Profiler Level 2 E%u S%u]\n"
        "        L2  shuffle     %.6lf | core sample  %.6lf | id remap  %.6lf\n"
        "        L2  graph copy  %.6lf | id copy      %.6lf | extract   %.6lf |"
        " feat copy %.6lf\n"
        "        L2  last layer sample time %.6lf | size %.6lf\n",
        type.c_str(), epoch, step, _step_buf[kLogL2ShuffleTime],
        _step_buf[kLogL2CoreSampleTime], _step_buf[kLogL2IdRemapTime],
        _step_buf[kLogL2GraphCopyTime], _step_buf[kLogL2IdCopyTime],
        _step_buf[kLogL2ExtractTime], _step_buf[kLogL2FeatCopyTime],
        _step_buf[kLogL2LastLayerTime], _step_buf[kLogL2LastLayerSize]);
  } else */if (level >= 2) {
    printf(
        "    [%s Profiler Level 2 E%u S%u]\n"
        "        L2  shuffle     %.6lf | core sample  %.6lf | "
        "id remap        %.6lf\n"
        "        L2  graph copy  %.6lf | id copy      %.6lf | "
        "cache feat copy %.6lf\n"
        "        L2  last layer sample time %.6lf | size %.6lf\n",
        type.c_str(), epoch, step, _step_buf[kLogL2ShuffleTime],
        _step_buf[kLogL2CoreSampleTime], _step_buf[kLogL2IdRemapTime],
        _step_buf[kLogL2GraphCopyTime], _step_buf[kLogL2IdCopyTime],
        _step_buf[kLogL2CacheCopyTime],
        _step_buf[kLogL2LastLayerTime], _step_buf[kLogL2LastLayerSize]);
  }

  /*if (level >= 3 && !RunConfig::UseGPUCache()) {
    printf(
        "     [%s Profiler Level 3 E%u S%u]\n"
        "        L3  khop sample coo  %.6lf | khop sort coo     %.6lf | "
        "khop count edge   %.6lf | khop compact edge %.6lf\n"
        "        L3  walk sample coo  %.6lf | walk topk total   %.6lf | "
        "walk topk step1   %.6lf | walk topk step2   %.6lf\n"
        "        L3  walk topk step3  %.6lf | walk topk step4   %.6lf | "
        "walk topk step5   %.6lf\n"
        "        L3  walk topk step6  %.6lf | walk topk step7   %.6lf\n"
        "        L3  remap unique     %.6lf | remap populate    %.6lf | "
        "remap mapnode     %.6lf | remap mapedge     %.6lf\n",
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
        _step_buf[kLogL3RemapFillUniqueTime],
        _step_buf[kLogL3RemapPopulateTime], _step_buf[kLogL3RemapMapNodeTime],
        _step_buf[kLogL3RemapMapEdgeTime]);
  } else */if (level >= 3) {
    printf(
        "    [%s Profiler Level 3 E%u S%u]\n"
        "        L3  khop sample coo  %.6lf | khop sort coo      %.6lf | "
        "khop count edge     %.6lf | khop compact edge %.6lf\n"
        "        L3  walk sample coo  %.6lf | walk topk total    %.6lf | "
        "walk topk step1     %.6lf | walk topk step2   %.6lf\n"
        "        L3  walk topk step3  %.6lf | walk topk step4    %.6lf | "
        "walk topk step5     %.6lf\n"
        "        L3  walk topk step6  %.6lf | walk topk step7    %.6lf\n"
        "        L3  remap unique     %.6lf | remap populate     %.6lf | "
        "remap mapnode       %.6lf | remap mapedge     %.6lf\n"
        "        L3  cache get_index  %.6lf | cache copy_index   %.6lf | "
        "cache extract_miss  %.6lf\n"
        "        L3  cache copy_miss  %.6lf | cache combine_miss %.6lf | "
        "cache combine cache %.6lf | cache combine remote %.6lf\n"
        "        L3  label extract  %.6lf\n",
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
        _step_buf[kLogL3RemapFillUniqueTime],
        _step_buf[kLogL3RemapPopulateTime], _step_buf[kLogL3RemapMapNodeTime],
        _step_buf[kLogL3RemapMapEdgeTime], _step_buf[kLogL3CacheGetIndexTime],
        _step_buf[KLogL3CacheCopyIndexTime],
        _step_buf[kLogL3CacheExtractMissTime],
        _step_buf[kLogL3CacheCopyMissTime],
        _step_buf[kLogL3CacheCombineMissTime],
        _step_buf[kLogL3CacheCombineCacheTime],
        _step_buf[kLogL3CacheCombineRemoteTime],
        _step_buf[kLogL3LabelExtractTime]);
  }
}

void Profiler::OutputEpoch(uint64_t epoch, std::string type) {
  printf(
      "  [%s Profiler E%u]\n"
      "      total %.6lf | sample %.6lf | copy %.6lf | train %.6lf\n",
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

#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for (size_t i = 0; i < num_input; ++i) {
    _epoch_cur_visit[input[i]] ++;
  }

  if (Engine::Get()->GetStepFromKey(key) == Engine::Get()->NumStep()-1) {
    std::vector<std::pair<size_t, IdType>> last_e_records, cur_e_records; // freq, nid
    std::vector<double> last_e_nid_to_rank(_last_visit.size()), cur_e_nid_to_rank(_last_visit.size()); // nid to rank
    last_e_records.resize(_last_visit.size());
    cur_e_records.resize(_last_visit.size());
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
    for (size_t i = 0; i < _last_visit.size(); i++) {
      last_e_records[i].first = _epoch_last_visit[i];
      last_e_records[i].second = i;
      cur_e_records[i].first = _epoch_cur_visit[i];
      cur_e_records[i].second = i;
    }
#ifdef __linux__
    __gnu_parallel::sort(last_e_records.begin(), last_e_records.end(),
                         std::greater<std::pair<size_t, IdType>>());
    __gnu_parallel::sort(cur_e_records.begin(), cur_e_records.end(),
                         std::greater<std::pair<size_t, IdType>>());
#else
    std::sort(last_e_records.begin(), last_e_records.end(),
              std::greater<std::pair<size_t, IdType>>());
    std::sort(cur_e_records.begin(), cur_e_records.end(),
              std::greater<std::pair<size_t, IdType>>());
#endif
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
    for (size_t rank = 0; rank < _last_visit.size(); rank++) {
      last_e_nid_to_rank[last_e_records[rank].second] = rank / (double)_last_visit.size() * 100;
      cur_e_nid_to_rank[cur_e_records[rank].second] = rank / (double)_last_visit.size() * 100;
    }

    size_t total_cur_e_top_K_freq = 0;
    size_t e_min_freq_both_e_top_K = 0;
#pragma omp parallel for num_threads(RunConfig::omp_thread_num) reduction(+ : e_min_freq_both_e_top_K, total_cur_e_top_K_freq)
    for (size_t i = 0; i < _epoch_last_visit.size(); ++i) {
      if (cur_e_nid_to_rank[i] < 10) {
        total_cur_e_top_K_freq += _epoch_cur_visit[i];
      }
      if (_epoch_cur_visit[i]) {
        if (last_e_nid_to_rank[i] < 10 && cur_e_nid_to_rank[i] < 10) {
          e_min_freq_both_e_top_K += Min(_epoch_last_visit[i], _epoch_cur_visit[i]);
        }
      }
    }
    _epoch_similarity[Engine::Get()->GetEpochFromKey(key)] = e_min_freq_both_e_top_K/(double)total_cur_e_top_K_freq*100;
    LOG(WARNING) << "top K min freq is both epoch's topK 10%: " 
                 << e_min_freq_both_e_top_K << "/" 
                 << total_cur_e_top_K_freq << "="
                 << e_min_freq_both_e_top_K/(double)total_cur_e_top_K_freq*100;
    // clean up
    _epoch_last_visit.swap(_epoch_cur_visit);
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
    for (size_t i = 0; i < _epoch_cur_visit.size(); ++i) {
      _epoch_cur_visit[i] = 0;
    }
  }

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
  // ofs0: frequency histogram: frequency per epoch of top 0~100% nodes
  // ofs1: binary file that looks like a cache file. can be used as cache
  // ofs2: optimal cache hit under different cache rate
  // ofs3: binary file containing each node's frequency

  std::ofstream ofs0(Constant::kNodeAccessFrequencyFile + GetTimeString() +
                         Constant::kNodeAccessFileSuffix,
                     std::ofstream::out | std::ofstream::trunc);
  std::ofstream ofs1(Constant::kNodeAccessOptimalCacheBinFile + GetTimeString() +
                         Constant::kNodeAccessFileSuffix,
                     std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
  std::ofstream ofs2(Constant::kNodeAccessOptimalCacheHitFile + GetTimeString() +
                         Constant::kNodeAccessFileSuffix,
                     std::ofstream::out | std::ofstream::trunc);
  std::ofstream ofs3(Constant::kNodeAccessOptimalCacheFreqBinFile + GetTimeString() +
                         Constant::kNodeAccessFileSuffix,
                     std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
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
    IdType frequency = p.first;
    float avg_frequency = frequency / static_cast<double>(Engine::Get()->NumEpoch());
    IdType nodeid = p.second;
    ofs1.write(reinterpret_cast<char*>(&nodeid), sizeof(IdType));
    ofs3.write(reinterpret_cast<char*>(&avg_frequency), sizeof(float));
    frequency_sum += frequency;
    p.first = frequency_sum;
  }

  for (int cache_rate = 0; cache_rate <= 100; cache_rate++) {
    int idx = ((uint64_t)cache_rate * num_nodes - 1) / 100;
    if (cache_rate == 0) idx = 0;
    ofs0 << cache_rate << "\t" << ((idx == 0) ? (double)0 : (records[idx].first - records[idx-1].first) / (double)Engine::Get()->NumEpoch()) << "\n";
    double hit_rate = ((idx == 0) ? 0 : records[idx].first / static_cast<double>(frequency_sum));
    ofs2 << cache_rate << "\t" << hit_rate << "\n";
  }

  ofs0.close();
  ofs1.close();
  ofs2.close();
  ofs3.close();

  double similarity_sum = 0;
  for (size_t e = 1; e < _epoch_similarity.size(); e++) {
    similarity_sum += _epoch_similarity[e];
  }
  printf("test_result:node_access:epoch_similarity=%lf\n", similarity_sum / (_epoch_similarity.size() - 1));
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
