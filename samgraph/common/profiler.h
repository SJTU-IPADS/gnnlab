#ifndef SAMGRAPH_PROFILER_H
#define SAMGRAPH_PROFILER_H

#include <cstdint>
#include <string>
#include <vector>

#include "common.h"

namespace samgraph {
namespace common {

enum LogItem {
  // L1
  kLogL1NumSample = 0,
  kLogL1NumNode,
  kLogL1SampleTime,
  kLogL1CopyTime,
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
  kLogL3SampleCooTime,
  kLogL3SampleSortCooTime,
  kLogL3SampleCountEdgeTime,
  kLogL3SampleCompactEdgesTime,
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
  kNumLogItems
};

struct LogData {
  std::vector<double> vals;
  double sum;
  size_t cnt;
  std::vector<bool> bitmap;

  LogData();
};

class Profiler {
 public:
  Profiler();
  void Log(uint64_t key, LogItem item, double value);
  void LogAdd(uint64_t key, LogItem item, double value);

  void ReportStep(uint64_t epoch, uint64_t step);
  void ReportStepAverage(uint64_t epoch, uint64_t step);
  void ReportEpoch(uint64_t epoch);
  void ReportEpochAverage(uint64_t epoch);

  void LogNodeAccess(uint64_t key, const IdType *input, size_t num_input);
  void ReportNodeAccess();

  static Profiler &Get();

 private:
  void Output(uint64_t key, std::string type);

  std::vector<LogData> _data;
  std::vector<double> _buf;

  std::vector<size_t> _node_access;
  std::vector<int> _last_visit;
  std::vector<size_t> _similarity;
};

}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_PROFILER_H
