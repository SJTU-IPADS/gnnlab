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
  kLogL1TrainTime,
  kLogL1FeatureBytes,
  kLogL1LabelBytes,
  kLogL1IdBytes,
  kLogL1GraphBytes,
  kLogL1MissBytes,
  // L2
  kLogL2ShuffleTime,
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
  kLogL3RandomWalkTopKStep12Time,
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

class Profiler {
 public:
  Profiler();
  void LogStep(uint64_t key, LogStepItem item, double val);
  void LogStepAdd(uint64_t key, LogStepItem item, double val);
  void LogEpochAdd(uint64_t key, LogEpochItem item, double val);

  double GetLogStepValue(uint64_t key, LogStepItem item);
  double GetLogEpochValue(uint64_t epoch, LogEpochItem item);

  void ReportStep(uint64_t epoch, uint64_t step);
  void ReportStepAverage(uint64_t epoch, uint64_t step);
  void ReportEpoch(uint64_t epoch);
  void ReportEpochAverage(uint64_t epoch);

  void LogNodeAccess(uint64_t key, const IdType *input, size_t num_input);
  void ReportNodeAccess();

  static Profiler &Get();

 private:
  void OutputStep(uint64_t key, std::string type);
  void OutputEpoch(uint64_t epoch, std::string type);

  std::vector<LogData> _step_data;
  std::vector<double> _step_buf;
  std::vector<LogData> _epoch_data;
  std::vector<double> _epoch_buf;

  // for node access
  std::vector<size_t> _node_access;
  std::vector<int> _last_visit;
  std::vector<size_t> _similarity;
};

}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_PROFILER_H
