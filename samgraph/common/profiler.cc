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

namespace samgraph {
namespace common {

LogData::LogData() {
  size_t num_logs = Engine::Get()->NumEpoch() * Engine::Get()->NumStep();
  vals.resize(num_logs);
  bitmap.resize(num_logs);
  sum = 0;
  cnt = 0;
}

Profiler::Profiler() {
  size_t num_items = static_cast<size_t>(kLogNumItemsNotARealValue);
  size_t num_logs = Engine::Get()->NumEpoch() * Engine::Get()->NumStep();
  _data.resize(num_items);
  _output_buf.resize(num_logs);
  _node_access.resize(Engine::Get()->GetGraphDataset()->num_node, 0);
  _last_visit.resize(Engine::Get()->GetGraphDataset()->num_node, 0);
  _similarity.resize(num_logs);
}

void Profiler::Log(uint64_t key, LogItem item, double val) {
  int item_idx = static_cast<int>(item);
  _data[item_idx].vals[key] = val;
  _data[item_idx].sum += val;
  _data[item_idx].cnt = _data[item_idx].bitmap[key] ? _data[item_idx].cnt
                                                    : _data[item_idx].cnt + 1;
  _data[item_idx].bitmap[key] = true;
}

void Profiler::LogAdd(uint64_t key, LogItem item, double val) {
  int item_idx = static_cast<int>(item);
  _data[item_idx].vals[key] += val;
  _data[item_idx].sum += val;
  _data[item_idx].cnt = _data[item_idx].bitmap[key] ? _data[item_idx].cnt
                                                    : _data[item_idx].cnt + 1;
  _data[item_idx].bitmap[key] = true;
}

void Profiler::LogNodeAccess(uint64_t key, const IdType *input,
                             size_t num_input) {
#pragma omp parallel for num_threads(RunConfig::kOMPThreadNum)
  for (size_t i = 0; i < num_input; ++i) {
    _node_access[input[i]]++;
  }

  size_t similarity_count = 0;
#pragma omp parallel for num_threads(RunConfig::kOMPThreadNum) reduction(+:similarity_count)
  for (size_t i = 0; i < num_input; ++i) {
    if (_last_visit[input[i]]) {
      similarity_count++;
    }
  }

#pragma omp parallel for num_threads(RunConfig::kOMPThreadNum)
  for (size_t i = 0; i < _last_visit.size(); ++i) {
    _last_visit[i] = 0;
  }

#pragma omp parallel for num_threads(RunConfig::kOMPThreadNum)
  for (size_t i = 0; i < num_input; ++i) {
    _last_visit[input[i]] = 1;
  }

  _similarity[key] = similarity_count;
}

void Profiler::Report(uint64_t key) {
  size_t num_items = static_cast<size_t>(kLogNumItemsNotARealValue);
  for (size_t i = 0; i < num_items; i++) {
    _output_buf[i] = _data[i].vals[key];
  }
  Output(key, "");
}

void Profiler::ReportAverage(uint64_t key) {
  size_t num_items = static_cast<size_t>(kLogNumItemsNotARealValue);
  for (size_t i = 0; i < num_items; i++) {
    double sum = _data[i].sum - _data[i].vals[0];
    size_t cnt = _data[i].cnt <= 1 ? 1 : _data[i].cnt - 1;
    _output_buf[i] = sum / cnt;
  }

  Output(key, "avg");
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
  double count_percentage_prefix_sum = 0;
  // access's prefix sum
  double access_percentage_prefix_sum = 0;

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
          std::min(min_indegree_map[frequency], in_degrees[nodeid]);
      max_indegree_map[frequency] =
          std::max(max_indegree_map[frequency], in_degrees[nodeid]);
      sum_outdegree_map[frequency] += out_degrees[nodeid];
      min_outdegree_map[frequency] =
          std::min(min_outdegree_map[frequency], out_degrees[nodeid]);
      max_outdegree_map[frequency] =
          std::max(min_outdegree_map[frequency], out_degrees[nodeid]);
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
    double count_percentage = static_cast<double>(count) / num_nodes;
    count_percentage_prefix_sum += count_percentage;

    size_t access = frequency * count;
    double access_percentage = static_cast<double>(access) / access_sum;
    access_percentage_prefix_sum += access_percentage;

    double average_indegree = static_cast<double>(sum_indegree_map[frequency]) /
                              static_cast<double>(count);
    double average_outdegree =
        static_cast<double>(sum_outdegree_map[frequency]) /
        static_cast<double>(count);

    ofs1 << frequency << " " << count << " " << count_percentage << " "
         << count_percentage_prefix_sum << " " << access << " "
         << access_percentage << " " << access_percentage_prefix_sum << " "
         << min_indegree_map[frequency] << " " << average_indegree << " "
         << max_indegree_map[frequency] << " " << min_outdegree_map[frequency]
         << " " << average_outdegree << " " << max_outdegree_map[frequency]
         << "\n";
  }

  for (size_t i = 0; i < _similarity.size(); i++) {
    double similarity_percentage =
        _similarity[i] / _data[kLogL1NumNode].vals[i];
    ofs2 << i << " " << _data[kLogL1NumNode].vals[i] << " " << _similarity[i]
         << " " << similarity_percentage << "\n";
  }

  ofs0.close();
  ofs1.close();
  ofs2.close();
}

Profiler &Profiler::Get() {
  static Profiler inst;
  return inst;
}

void Profiler::Output(uint64_t key, std::string tag) {
  uint64_t epoch = Engine::Get()->GetEpochFromKey(key);
  uint64_t step = Engine::Get()->GetStepFromKey(key);

  std::string env_level = GetEnv(Constant::kEnvProfileLevel);

  int level = 0;
  if (env_level == "1") {
    level = 1;
  } else if (env_level == "2") {
    level = 2;
  } else if (env_level == "3") {
    level = 3;
  }

  if (level >= 1 && !RunConfig::UseGPUCache()) {
    printf(
        "  [Profile L1-%s %lu %lu]"
        " sample %.4lf |"
        " copy %.4lf |"
        " feature nbytes %s |"
        " label nbytes %s |"
        " id nbytes %s |"
        " graph nbytes %s \n",
        tag.c_str(), epoch, step, _output_buf[kLogL1SampleTime],
        _output_buf[kLogL1CopyTime],
        ToReadableSize(_output_buf[kLogL1FeatureBytes]).c_str(),
        ToReadableSize(_output_buf[kLogL1LabelBytes]).c_str(),
        ToReadableSize(_output_buf[kLogL1IdBytes]).c_str(),
        ToReadableSize(_output_buf[kLogL1GraphBytes]).c_str());
  } else {
    printf(
        "  [Profile L1-%s %lu %lu]"
        " sample %.4lf |"
        " copy %.4lf |"
        " feature nbytes %s |"
        " label nbytes %s |"
        " id nbytes %s |"
        " graph nbytes %s |"
        " miss nbytes %s\n",
        tag.c_str(), epoch, step, _output_buf[kLogL1SampleTime],
        _output_buf[kLogL1CopyTime],
        ToReadableSize(_output_buf[kLogL1FeatureBytes]).c_str(),
        ToReadableSize(_output_buf[kLogL1LabelBytes]).c_str(),
        ToReadableSize(_output_buf[kLogL1IdBytes]).c_str(),
        ToReadableSize(_output_buf[kLogL1GraphBytes]).c_str(),
        ToReadableSize(_output_buf[kLogL1MissBytes]).c_str());
  }

  if (level >= 2 && !RunConfig::UseGPUCache()) {
    printf(
        "  [Profile L2-%s %lu %lu]"
        " shuffle %.4lf |"
        " core sample %.4lf |"
        " id remap %.4lf |"
        " graph copy %.4lf |"
        " id copy %.4lf |"
        " extract %.4lf |"
        " feat copy %.4lf\n",
        tag.c_str(), epoch, step, _output_buf[kLogL2ShuffleTime],
        _output_buf[kLogL2CoreSampleTime], _output_buf[kLogL2IdRemapTime],
        _output_buf[kLogL2GraphCopyTime], _output_buf[kLogL2IdCopyTime],
        _output_buf[kLogL2ExtractTime], _output_buf[kLogL2FeatCopyTime]);
  } else if (level >= 2) {
    printf(
        "  [Profile L2-%s %lu %lu]"
        " shuffle %.4lf |"
        " core sample %.4lf |"
        " id remap %.4lf |"
        " graph copy %.4lf |"
        " id copy %.4lf |"
        " cache feat copy %.4lf\n",
        tag.c_str(), epoch, step, _output_buf[kLogL2ShuffleTime],
        _output_buf[kLogL2CoreSampleTime], _output_buf[kLogL2IdRemapTime],
        _output_buf[kLogL2GraphCopyTime], _output_buf[kLogL2IdCopyTime],
        _output_buf[kLogL2CacheCopyTime]);
  }

  if (level >= 3 && !RunConfig::UseGPUCache()) {
    printf(
        "  [Profile L3-%s %lu %lu]"
        " sample coo %.4lf |"
        " count edge %.4lf |"
        " compact edge %.4lf |"
        " remap populate %.4lf |"
        " remap mapnode %.4lf |"
        " remap mapedge %.4lf\n",
        tag.c_str(), epoch, step, _output_buf[kLogL3SampleCooTime],
        _output_buf[kLogL3SampleCountEdgeTime],
        _output_buf[kLogL3SampleCompactEdgesTime],
        _output_buf[kLogL3RemapPopulateTime],
        _output_buf[kLogL3RemapMapNodeTime],
        _output_buf[kLogL3RemapMapEdgeTime]);
  } else if (level >= 3) {
    printf(
        "  [Profile L3-%s %lu %lu]"
        " sample coo %.4lf |"
        " count edge %.4lf |"
        " compact edge %.4lf |"
        " remap populate %.4lf |"
        " remap mapnode %.4lf |"
        " remap mapedge %.4lf |"
        " cache get_index %.4lf |"
        " cache copy_index %.4lf |"
        " cache extract_miss %.4lf |"
        " cache copy_miss %.4lf |"
        " cache combine_miss %.4lf |"
        " cache combine cache %.4lf\n",
        tag.c_str(), epoch, step, _output_buf[kLogL3SampleCooTime],
        _output_buf[kLogL3SampleCountEdgeTime],
        _output_buf[kLogL3SampleCompactEdgesTime],
        _output_buf[kLogL3RemapPopulateTime],
        _output_buf[kLogL3RemapMapNodeTime],
        _output_buf[kLogL3RemapMapEdgeTime],
        _output_buf[kLogL3CacheGetIndexTime],
        _output_buf[KLogL3CacheCopyIndexTime],
        _output_buf[kLogL3CacheExtractMissTime],
        _output_buf[kLogL3CacheCopyMissTime],
        _output_buf[kLogL3CacheCombineMissTime],
        _output_buf[kLogL3CacheCombineCacheTime]);
  }
}

}  // namespace common
}  // namespace samgraph
