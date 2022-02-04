#ifndef SAMGRAPH_PROFILER_H
#define SAMGRAPH_PROFILER_H

#include <cstdint>
#include <string>
#include <vector>

#include "common.h"

namespace samgraph {
namespace common {

enum LogInitItem {
  // L1
  kLogInitL1Common = 0,
  kLogInitL1Sampler, // pinmem, copy graph, *shuff, *hashtable, *random, *freq, queue, presmaple, cachetable
  kLogInitL1Trainer,
  // L2
  kLogInitL2LoadDataset,
  kLogInitL2DistQueue,
  kLogInitL2Presample,
  kLogInitL2InternalState,
  kLogInitL2BuildCache,
  // L3
  kLogInitL3LoadDatasetMMap,
  kLogInitL3LoadDatasetCopy,
  kLogInitL3DistQueueAlloc,
  kLogInitL3DistQueuePin,
  kLogInitL3DistQueuePush,
  kLogInitL3PresampleInit,
  kLogInitL3PresampleSample,
  kLogInitL3PresampleCopy,
  kLogInitL3PresampleCount,
  kLogInitL3PresampleSort,
  kLogInitL3PresampleReset,
  kLogInitL3PresampleGetRank,
  kLogInitL3InternalStateCreateCtx,
  kLogInitL3InternalStateCreateStream,
  kNumLogInitItems,
};

enum LogStepItem {
  // L1
  kLogL1NumSample = 0,
  kLogL1NumNode,
  kLogL1SampleTime,
  kLogL1SendTime,
  kLogL1RecvTime,
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
  kLogL3KHopPartitionSampleLoadTime,
  kLogL3KHopPartitionSampleTime,
  // Number of items
  kNumLogStepItems
};

enum LogEpochItem {
  kLogEpochSampleTime = 0,
  KLogEpochSampleGetCacheMissIndexTime,  // for arch5
  kLogEpochSampleSendTime,               // for arch5
  kLogEpochSampleTotalTime,              // for arch5
  kLogEpochCopyTime,
  kLogEpochConvertTime,
  kLogEpochTrainTime,
  kLogEpochTotalTime,
  kLogEpochFeatureBytes,
  kLogEpochMissBytes,
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
  void ResetStepEpoch();
  void LogInit(LogInitItem item, double val);
  void LogInitAdd(LogInitItem item, double val);
  void LogStep(uint64_t key, LogStepItem item, double val);
  void LogStepAdd(uint64_t key, LogStepItem item, double val);
  void LogEpochAdd(uint64_t key, LogEpochItem item, double val);

  inline void TraceStepBegin(uint64_t key, TraceItem item, uint64_t us) { _step_trace[item].events[key].begin = us; }
  inline void TraceStepEnd(uint64_t key, TraceItem item, uint64_t us) { _step_trace[item].events[key].end = us; }

  double GetLogInitValue(LogInitItem item);
  double GetLogStepValue(uint64_t key, LogStepItem item);
  double GetLogEpochValue(uint64_t epoch, LogEpochItem item);

  void ReportInit();
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

  std::vector<LogData> _init_data;
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
  std::vector<int> _epoch_last_visit;
  std::vector<int> _epoch_cur_visit;
  std::vector<size_t> _epoch_similarity;
};

}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_PROFILER_H
