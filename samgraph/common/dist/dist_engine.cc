#include "dist_engine.h"

#include "../constant.h"
#include "../device.h"
#include "../logging.h"
#include "../run_config.h"
#include "../timer.h"

// TODO: decide CPU or GPU to shuffling, sampling and id remapping
#include "cpu_hashtable0.h"
#include "cpu_hashtable1.h"
#include "cpu_hashtable2.h"
#include "cpu_loops.h"

namespace samgraph {
namespace common {
namespace dist {

DistEngine::DistEngine() {
  _initialize = false;
  _should_shutdown = false;
}

void DistEngine::Init() {
  if (_initialize) {
    return;
  }

  // FIXME: _sampler_ctx and _trainer_ctx initialized later
  // _sampler_ctx = RunConfig::sampler_ctx;
  // _trainer_ctx = RunConfig::trainer_ctx;
  _dataset_path = RunConfig::dataset_path;
  _batch_size = RunConfig::batch_size;
  _fanout = RunConfig::fanout;
  _num_epoch = RunConfig::num_epoch;
  _joined_thread_cnt = 0;

  // Check whether the ctx configuration is allowable
  ArchCheck();

  // Load the target graph data
  LoadGraphDataset();

  LOG(DEBUG) << "Finished pre-initialization";
}

// TODO: add sample_init for sampling process


// TODO: add train_init for extracting and training process


void DistEngine::Start() {
  LOG(FATAL) << "DistEngine needs not implement the Start function!!!";
}

void DistEngine::Shutdown() {
  LOG(FATAL) << "DistEngine needs not implement the Shutdown function!!!";
}

// TODO: implement it!
//       and split the sampling and extracting
void DistEngine::RunSampleOnce() {

  LOG(DEBUG) << "RunSampleOnce finished.";
}

void CPUEngine::ArchCheck() {
  CHECK_EQ(RunConfig::run_arch, kArch5);
}

std::unordered_map<std::string, Context> DistEngine::GetGraphFileCtx() {
  std::unordered_map<std::string, Context> ret;

  ret[Constant::kIndptrFile] = MMAP();
  ret[Constant::kIndicesFile] = MMAP();
  ret[Constant::kFeatFile] = MMAP();
  ret[Constant::kLabelFile] = MMAP();
  ret[Constant::kTrainSetFile] = MMAP();
  ret[Constant::kTestSetFile] = MMAP();
  ret[Constant::kValidSetFile] = MMAP();
  ret[Constant::kProbTableFile] = MMAP();
  ret[Constant::kAliasTableFile] = MMAP();
  ret[Constant::kInDegreeFile] = MMAP();
  ret[Constant::kOutDegreeFile] = MMAP();
  ret[Constant::kCacheByDegreeFile] = MMAP();
  ret[Constant::kCacheByHeuristicFile] = MMAP();

  return ret;
}

}  // namespace dist
}  // namespace common
}  // namespace samgraph
