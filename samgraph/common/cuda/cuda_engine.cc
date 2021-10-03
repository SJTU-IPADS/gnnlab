#include "cuda_engine.h"

#include <chrono>
#include <cstdlib>
#include <numeric>

#include "../common.h"
#include "../constant.h"
#include "../device.h"
#include "../logging.h"
#include "../run_config.h"
#include "cuda_common.h"
#include "cuda_loops.h"
#include "pre_sampler.h"
#include "../profiler.h"
#include "../timer.h"

namespace samgraph {
namespace common {
namespace cuda {

namespace {
size_t get_cuda_used(Context ctx) {
  size_t free, used;
  cudaSetDevice(ctx.device_id);
  cudaMemGetInfo(&free, &used);
  return used - free;
}
#define LOG_MEM_USAGE(LEVEL, title) \
  {\
    auto _sampler_ctx = RunConfig::sampler_ctx;\
    auto _trainer_ctx = RunConfig::trainer_ctx;\
    auto _gpu_device = Device::Get(_sampler_ctx); \
    LOG(LEVEL) << (title) << ", data alloc: sampler " << ToReadableSize(_gpu_device->DataSize(_sampler_ctx))      << ", trainer " << ToReadableSize(_gpu_device->DataSize(_trainer_ctx));\
    LOG(LEVEL) << (title) << ", workspace : sampler " << ToReadableSize(_gpu_device->WorkspaceSize(_sampler_ctx)) << ", trainer " << ToReadableSize(_gpu_device->WorkspaceSize(_trainer_ctx));\
    LOG(LEVEL) << (title) << ", total: sampler "      << ToReadableSize(_gpu_device->TotalSize(_sampler_ctx))     << ", trainer " << ToReadableSize(_gpu_device->TotalSize(_trainer_ctx));\
    LOG(LEVEL) << "cuda usage: sampler " << ToReadableSize(get_cuda_used(_sampler_ctx)) << ", trainer " << ToReadableSize(get_cuda_used(_trainer_ctx));\
  }
}

GPUEngine::GPUEngine() {
  _initialize = false;
  _should_shutdown = false;
}

void GPUEngine::Init() {
  if (_initialize) {
    return;
  }
  Timer t_init;
  _sampler_ctx = RunConfig::sampler_ctx;
  _trainer_ctx = RunConfig::trainer_ctx;
  _dataset_path = RunConfig::dataset_path;
  _batch_size = RunConfig::batch_size;
  _fanout = RunConfig::fanout;
  _num_epoch = RunConfig::num_epoch;
  _joined_thread_cnt = 0;

  // Check whether the ctx configuration is allowable
  ArchCheck();

  Timer t_cuda_context;
  // Load the target graph data
  LOG_MEM_USAGE(WARNING, "before load dataset");
  double time_cuda_context = t_cuda_context.Passed();

  Timer tl;
  LoadGraphDataset();
  double time_load_graph_dataset = tl.Passed();
  LOG_MEM_USAGE(INFO, "after load dataset");

  // Create CUDA streams
  Timer t_create_stream;
  _sample_stream = Device::Get(_sampler_ctx)->CreateStream(_sampler_ctx);
  _sampler_copy_stream = Device::Get(_sampler_ctx)->CreateStream(_sampler_ctx);
  _trainer_copy_stream = Device::Get(_trainer_ctx)->CreateStream(_trainer_ctx);

  Device::Get(_sampler_ctx)->StreamSync(_sampler_ctx, _sample_stream);
  Device::Get(_sampler_ctx)->StreamSync(_sampler_ctx, _sampler_copy_stream);
  Device::Get(_trainer_ctx)->StreamSync(_trainer_ctx, _trainer_copy_stream);
  double time_create_stream = t_create_stream.Passed();

  Timer t_sam_interal_state;
  _shuffler =
      new GPUShuffler(_dataset->train_set, _num_epoch, _batch_size, false);
  _num_step = _shuffler->NumStep();

#ifndef SXN_NAIVE_HASHMAP
  _hashtable = new OrderedHashTable(
      PredictNumNodes(_batch_size, _fanout, _fanout.size()), _sampler_ctx);
#else
  _hashtable = new OrderedHashTable(
      _dataset->num_node, _sampler_ctx, 1);
#endif
  LOG_MEM_USAGE(INFO, "after create hashtable");

  // Create CUDA random states for sampling
  _random_states = new GPURandomStates(RunConfig::sample_type, _fanout,
                                       _batch_size, _sampler_ctx);

  LOG_MEM_USAGE(INFO, "after create states");
  if (RunConfig::sample_type == kRandomWalk) {
    size_t max_nodes =
        PredictNumNodes(_batch_size, _fanout, _fanout.size() - 1);
    size_t edges_per_node =
        RunConfig::num_random_walk * RunConfig::random_walk_length;
    _frequency_hashmap =
        new FrequencyHashmap(max_nodes, edges_per_node, _sampler_ctx);
  } else {
    _frequency_hashmap = nullptr;
  }

  // Create queues
  for (int i = 0; i < QueueNum; i++) {
    LOG(DEBUG) << "Create task queue" << i;
    _queues.push_back(new TaskQueue(RunConfig::max_sampling_jobs));
  }
  _graph_pool = new GraphPool(RunConfig::max_copying_jobs);
  double time_sam_internal_state = t_sam_interal_state.Passed();

  double presample_time = 0;
  double build_cache_time = 0;
  if (RunConfig::UseGPUCache()) {
    switch (RunConfig::cache_policy) {
      case kCacheByPreSampleStatic: 
      case kCacheByPreSample: {
        Timer tp;
        PreSampler::SetSingleton(new PreSampler(_dataset->num_node, NumStep()));
        PreSampler::Get()->DoPreSample();
        _dataset->ranking_nodes = PreSampler::Get()->GetRankNode();
        presample_time = tp.Passed();
        break;
      }
      default: ;
    }
    Timer tc;
    _cache_manager = new GPUCacheManager(
        _sampler_ctx, _trainer_ctx, _dataset->feat->Data(),
        _dataset->feat->Type(), _dataset->feat->Shape()[1],
        static_cast<const IdType*>(_dataset->ranking_nodes->Data()),
        _dataset->num_node, RunConfig::cache_percentage);
    build_cache_time = tc.Passed();
    _dynamic_cache_manager = nullptr;
  } else if (RunConfig::UseDynamicGPUCache()) {
    Timer tc;
    _dynamic_cache_manager = new GPUDynamicCacheManager(
      _sampler_ctx, _trainer_ctx, _dataset->feat->Data(),
      _dataset->feat->Type(), _dataset->feat->Shape()[1],
      _dataset->num_node);
    build_cache_time = tc.Passed();
    _cache_manager = nullptr;
  } else {
    _cache_manager = nullptr;
    _dynamic_cache_manager = nullptr;
  }

  double time_init = t_init.Passed();
  Profiler::Get().LogInit(kLogInitL1Common, time_init);
  Profiler::Get().LogInit(kLogInitL2InternalState, time_sam_internal_state);
  Profiler::Get().LogInit(kLogInitL2Presample, presample_time);
  Profiler::Get().LogInit(kLogInitL2BuildCache, build_cache_time);
  Profiler::Get().LogInit(kLogInitL2LoadDataset, time_load_graph_dataset);

  Profiler::Get().LogInit(kLogInitL3InternalStateCreateCtx, time_cuda_context);
  Profiler::Get().LogInit(kLogInitL3InternalStateCreateStream, time_create_stream);

  LOG_MEM_USAGE(WARNING, "after build cache states");

  _initialize = true;
}

void GPUEngine::Start() {
  std::vector<LoopFunction> func;
  switch (RunConfig::run_arch) {
    case kArch1:
      func = GetArch1Loops();
      break;
    case kArch2:
      func = GetArch2Loops();
      break;
    case kArch3:
      func = GetArch3Loops();
      break;
    case kArch4:
      func = GetArch4Loops();
      break;
    default:
      // Not supported arch 0
      CHECK(0);
  }

  // Start background threads
  for (size_t i = 0; i < func.size(); i++) {
    _threads.push_back(new std::thread(func[i]));
  }
  LOG(DEBUG) << "Started " << func.size() << " background threads.";
}

void GPUEngine::Shutdown() {
  LOG_MEM_USAGE(WARNING, "end of train");
  if (_should_shutdown) {
    return;
  }

  _should_shutdown = true;
  int total_thread_num = _threads.size();

  while (!IsAllThreadFinish(total_thread_num)) {
    // wait until all threads joined
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }

  for (size_t i = 0; i < _threads.size(); i++) {
    _threads[i]->join();
    delete _threads[i];
    _threads[i] = nullptr;
  }

  // free queue
  for (size_t i = 0; i < QueueNum; i++) {
    if (_queues[i]) {
      delete _queues[i];
      _queues[i] = nullptr;
    }
  }

  Device::Get(_sampler_ctx)->StreamSync(_sampler_ctx, _sample_stream);
  Device::Get(_sampler_ctx)->FreeStream(_sampler_ctx, _sample_stream);
  Device::Get(_sampler_ctx)->StreamSync(_sampler_ctx, _sampler_copy_stream);
  Device::Get(_sampler_ctx)->FreeStream(_sampler_ctx, _sampler_copy_stream);

  Device::Get(_trainer_ctx)->StreamSync(_trainer_ctx, _trainer_copy_stream);
  Device::Get(_trainer_ctx)->FreeStream(_trainer_ctx, _trainer_copy_stream);

  delete _dataset;
  delete _shuffler;
  delete _graph_pool;
  delete _random_states;

  if (_cache_manager != nullptr) {
    delete _cache_manager;
  }

  if (_frequency_hashmap != nullptr) {
    delete _frequency_hashmap;
  }

  _dataset = nullptr;
  _shuffler = nullptr;
  _graph_pool = nullptr;
  _cache_manager = nullptr;
  _random_states = nullptr;
  _frequency_hashmap = nullptr;

  _threads.clear();
  _joined_thread_cnt = 0;
  _initialize = false;
  _should_shutdown = false;
}

void GPUEngine::RunSampleOnce() {
  switch (RunConfig::run_arch) {
    case kArch1:
      RunArch1LoopsOnce();
      break;
    case kArch2:
      RunArch2LoopsOnce();
      break;
    case kArch3:
      RunArch3LoopsOnce();
      break;
    case kArch4:
      RunArch4LoopsOnce();
      break;
    default:
      // Not supported arch 0
      CHECK(0);
  }
  LOG_MEM_USAGE(INFO, "after one batch");
}

void GPUEngine::ArchCheck() {
  CHECK_EQ(_sampler_ctx.device_type, kGPU);
  CHECK_EQ(_trainer_ctx.device_type, kGPU);

  switch (RunConfig::run_arch) {
    case kArch1:
      CHECK_EQ(_sampler_ctx.device_id, _trainer_ctx.device_id);
      break;
    case kArch2:
      CHECK_EQ(_sampler_ctx.device_id, _trainer_ctx.device_id);
      break;
    case kArch3:
      CHECK_NE(_sampler_ctx.device_id, _trainer_ctx.device_id);
      CHECK(!(RunConfig::UseGPUCache() && RunConfig::option_log_node_access));
      break;
    case kArch4:
      CHECK_NE(_sampler_ctx.device_id, _trainer_ctx.device_id);
      break;
    default:
      CHECK(0);
  }
}

std::unordered_map<std::string, Context> GPUEngine::GetGraphFileCtx() {
  std::unordered_map<std::string, Context> ret;

  ret[Constant::kIndptrFile] = _sampler_ctx;
  ret[Constant::kIndicesFile] = _sampler_ctx;
  ret[Constant::kTrainSetFile] = CPU();
  ret[Constant::kTestSetFile] = CPU();
  ret[Constant::kValidSetFile] = CPU();
  ret[Constant::kProbTableFile] = _sampler_ctx;
  ret[Constant::kAliasTableFile] = _sampler_ctx;
  ret[Constant::kInDegreeFile] = MMAP();
  ret[Constant::kOutDegreeFile] = MMAP();
  ret[Constant::kCacheByDegreeFile] = MMAP();
  ret[Constant::kCacheByHeuristicFile] = MMAP();
  ret[Constant::kCacheByDegreeHopFile] = MMAP();
  ret[Constant::kCacheByFakeOptimalFile] = MMAP();

  switch (RunConfig::run_arch) {
    case kArch1:
      ret[Constant::kFeatFile] = _sampler_ctx;
      ret[Constant::kLabelFile] = _sampler_ctx;
      break;
    case kArch2:
    case kArch3:
      ret[Constant::kFeatFile] = MMAP();
      ret[Constant::kLabelFile] =
          RunConfig::UseGPUCache() ? _trainer_ctx : MMAP();
      break;
    case kArch4:
      ret[Constant::kFeatFile] = MMAP();
      ret[Constant::kLabelFile] = 
          RunConfig::UseDynamicGPUCache() ? _trainer_ctx : MMAP();
      break;
    default:
      CHECK(0);
  }

  return ret;
}

}  // namespace cuda
}  // namespace common
}  // namespace samgraph
