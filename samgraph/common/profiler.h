#ifndef SAMGRAPH_PROFILER_H
#define SAMGRAPH_PROFILER_H

#include <cstdint>
#include <string>
#include <vector>

#include "common.h"

namespace samgraph {
namespace common {

enum LogStepItem {
  // L1
  kLogL1NumSample = 0,
  kLogL1NumNode,
  kLogL1SampleTime,
  kLogL1CopyTime,
  kLogL1ConvertTime,
  kLogL1TrainTime,
  kLogL1FeatureBytes,
  kLogL1LabelBytes,
  kLogL1IdBytes,
  kLogL1GraphBytes,
  kLogL1MissBytes,
  kLogL1PrefetchAdvanced,
  kLogL1GetNeighbourTime,
  // L2
  kLogL2ShuffleTime,
  kLogL2LastLayerTime,
  kLogL2LastLayerSize,
  kLogL2CoreSampleTime,
  kLogL2IdRemapTime,
  kLogL2GraphCopyTime,
  kLogL2IdCopyTime,
  kLogL2ExtractTime,
  kLogL2FeatCopyTime,
  kLogL2CacheCopyTime,
  // L3
  kLogL3KHopSampleCooTime,
  kLogL3KHopSampleSortCooTime,
  kLogL3KHopSampleCountEdgeTime,
  kLogL3KHopSampleCompactEdgesTime,
  kLogL3RandomWalkSampleCooTime,
  kLogL3RandomWalkTopKTime,
  kLogL3RandomWalkTopKStep1Time,
  kLogL3RandomWalkTopKStep2Time,
  kLogL3RandomWalkTopKStep3Time,
  kLogL3RandomWalkTopKStep4Time,
  kLogL3RandomWalkTopKStep5Time,
  kLogL3RandomWalkTopKStep6Time,
  kLogL3RandomWalkTopKStep7Time,
  kLogL3RandomWalkTopKStep8Time,
  kLogL3RandomWalkTopKStep9Time,
  kLogL3RandomWalkTopKStep10Time,
  kLogL3RandomWalkTopKStep11Time,
  kLogL3RemapFillUniqueTime,
  kLogL3RemapPopulateTime,
  kLogL3RemapMapNodeTime,
  kLogL3RemapMapEdgeTime,
  kLogL3CacheGetIndexTime,
  KLogL3CacheCopyIndexTime,
  kLogL3CacheExtractMissTime,
  kLogL3CacheCopyMissTime,
  kLogL3CacheCombineMissTime,
  kLogL3CacheCombineCacheTime,
  // Number of items
  kNumLogStepItems
};

enum LogEpochItem {
  kLogEpochSampleTime = 0,
  kLogEpochCopyTime,
  kLogEpochConvertTime,
  kLogEpochTrainTime,
  kLogEpochTotalTime,
  kNumLogEpochItems
};

struct LogData {
  std::vector<double> vals;
  double sum;
  size_t cnt;
  std::vector<bool> bitmap;

  LogData(size_t num_logs);
};

#define TRACE_TYPES( F ) \
  F(kL0Event_Train_Step) \
  F(kL1Event_Sample) \
  F(kL2Event_Sample_Shuffle) \
  F(kL2Event_Sample_Core) \
  F(kL2Event_Sample_IdRemap) \
  F(kL1Event_Copy) \
  F(kL2Event_Copy_Id) \
  F(kL2Event_Copy_Graph) \
  F(kL2Event_Copy_Extract) /*for non cache*/ \
  F(kL2Event_Copy_FeatCopy) /*for non cache*/\
  F(kL2Event_Copy_CacheCopy) /*for cache*/ \
  F(kL3Event_Copy_CacheCopy_GetIndex) \
  F(kL3Event_Copy_CacheCopy_CopyIndex) \
  F(kL3Event_Copy_CacheCopy_ExtractMiss) \
  F(kL3Event_Copy_CacheCopy_CopyMiss) \
  F(kL3Event_Copy_CacheCopy_CombineMiss) \
  F(kL3Event_Copy_CacheCopy_CombineCache) \
  F(kL1Event_Convert) \
  F(kL1Event_Train) 

#define F(name) name,
enum TraceItem {TRACE_TYPES( F ) kNumTraceItems };
#undef F

struct TraceEvent {
  uint64_t begin, end;

  TraceEvent();
};
struct TraceData {
  std::vector<TraceEvent> events;

  TraceData(size_t num_traces);
};

class Profiler {
 public:
  Profiler();
  void LogStep(uint64_t key, LogStepItem item, double val);
  void LogStepAdd(uint64_t key, LogStepItem item, double val);
  void LogEpochAdd(uint64_t key, LogEpochItem item, double val);

  inline void TraceStepBegin(uint64_t key, TraceItem item, uint64_t us) { _step_trace[item].events[key].begin = us; }
  inline void TraceStepEnd(uint64_t key, TraceItem item, uint64_t us) { _step_trace[item].events[key].end = us; }

  double GetLogStepValue(uint64_t key, LogStepItem item);
  double GetLogEpochValue(uint64_t epoch, LogEpochItem item);

  void ReportStep(uint64_t epoch, uint64_t step);
  void ReportStepAverage(uint64_t epoch, uint64_t step);
  void ReportEpoch(uint64_t epoch);
  void ReportEpochAverage(uint64_t epoch);

  void DumpTrace(std::ostream & of);

  void LogNodeAccess(uint64_t key, const IdType *input, size_t num_input);
  void ReportNodeAccess();
  void ReportNodeAccessSimple();

  void ReportPreSampleSimilarity();

  static Profiler &Get();

 private:
  void OutputStep(uint64_t key, std::string type);
  void OutputEpoch(uint64_t epoch, std::string type);

  std::vector<LogData> _step_data;
  std::vector<double> _step_buf;
  std::vector<LogData> _epoch_data;
  std::vector<double> _epoch_buf;

  // for trace
  std::vector<TraceData> _step_trace;
  // std::vector<TraceData> _epoch_trace;
  uint64_t _num_step;

  // for node access
  std::vector<size_t> _node_access;
  std::vector<int> _last_visit;
  std::vector<size_t> _similarity;
};

}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_PROFILER_H
