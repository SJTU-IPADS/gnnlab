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

#include "cuda_engine.h"

#include <chrono>
#include <cstdlib>
#include <numeric>
#include <parallel/algorithm>
#include <parallel/numeric>

#include "../common.h"
#include "../constant.h"
#include "../device.h"
#include "../dist/dist_shuffler_aligned.h"
#include "../logging.h"
#include "../profiler.h"
#include "../run_config.h"
#include "../timer.h"
#include "cuda_common.h"
#include "cuda_loops.h"
#include "pre_sampler.h"
#include "um_pre_sampler.h"

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
  CHECK(cusparseCreate(&_cusparse_handle) == CUSPARSE_STATUS_SUCCESS);
  CHECK(cusparseSetStream(_cusparse_handle, (cudaStream_t)_sample_stream) == CUSPARSE_STATUS_SUCCESS);

  Device::Get(_sampler_ctx)->StreamSync(_sampler_ctx, _sample_stream);
  Device::Get(_sampler_ctx)->StreamSync(_sampler_ctx, _sampler_copy_stream);
  Device::Get(_trainer_ctx)->StreamSync(_trainer_ctx, _trainer_copy_stream);
  double time_create_stream = t_create_stream.Passed();

  Timer t_sam_interal_state;
  if (RunConfig::run_arch == kArch7) {
    _shuffler = new dist::DistAlignedShuffler(_dataset->train_set, _num_epoch,
                                              _batch_size, RunConfig::worker_id,
                                              RunConfig::num_worker);
    _num_step = _shuffler->NumStep();
    _num_local_step = _shuffler->NumLocalStep();

  } else {
    _shuffler =
        new GPUShuffler(_dataset->train_set, _num_epoch, _batch_size, false);
    _num_step = _shuffler->NumStep();
  }

#ifndef SXN_NAIVE_HASHMAP
  _hashtable = new OrderedHashTable(
      PredictNumNodes(_batch_size, _fanout, _fanout.size()), _sampler_ctx, _sampler_copy_stream);
#else
  _hashtable = new OrderedHashTable(
      _dataset->num_node, _sampler_ctx, _sampler_copy_stream, 1);
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
  _negative_generator = new ArrayGenerator();

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

  // sort dataset(UM)
  LOG(INFO) << "unified_memory: " << RunConfig::unified_memory << " | "
            << "unified_memory_policy: " << static_cast<int>(RunConfig::unified_memory_policy);
  if(RunConfig::unified_memory) {
    Timer sort_um_tm;
    size_t num_nodes = _dataset->indptr->Shape()[0] - 1;
    size_t num_trainset = _dataset->train_set->Shape()[0];
    TensorPtr order;
    switch (RunConfig::unified_memory_policy)
    {
    case UMPolicy::kDegree: {
      // case 1: by degree
      LOG(INFO) << "sort um dataset by Degree";
      order = Tensor::FromMmap(
        _dataset_path + Constant::kCacheByDegreeFile,
        DataType::kI32, {num_nodes},
        CPU(), "order");
      break;
    }
    case UMPolicy::kTrainset: {
      // case 2: by train set
      LOG(INFO) << "sort um dataset by Trainset";
      char* is_trainset = static_cast<char*>(Device::Get(CPU())->AllocWorkspace(
        CPU(), sizeof(char) * num_nodes, Constant::kAllocNoScale));
      auto degree_order_ts = Tensor::FromMmap(
        _dataset_path + Constant::kCacheByDegreeFile,
        DataType::kI32, {num_nodes},
        CPU(), "order");
      auto degree_order = static_cast<const IdType*>(degree_order_ts->Data());
      order = Tensor::EmptyNoScale(DataType::kI32, {num_nodes}, CPU(), "");
      auto order_ptr = static_cast<IdType*>(order->MutableData());
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
      for(IdType i = 0; i < num_nodes; i++) {
        order_ptr[i] = i;
        is_trainset[i] = false;
      }
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
      for(IdType i = 0; i < num_trainset; i++) {
        auto trainset = static_cast<const IdType*>(_dataset->train_set->Data());
        is_trainset[trainset[i]] = true;
      }
      __gnu_parallel::sort(order_ptr, order_ptr + num_nodes, [&](IdType x, IdType y){
        return std::pair<IdType, IdType>{!is_trainset[x], degree_order[x]}
          < std::pair<IdType, IdType>{!is_trainset[y], degree_order[y]};
      });
      Device::Get(CPU())->FreeWorkspace(CPU(), is_trainset);
      break;
    }
    case UMPolicy::kRandom: {
      // case 3: by random
      LOG(INFO) << "sort um dataset by Random";
      order = Tensor::EmptyNoScale(DataType::kI32, {num_nodes}, CPU(), "order");
      auto order_ptr = static_cast<IdType*>(order->MutableData());
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
      for(IdType i = 0; i < num_nodes; i++) {
        order_ptr[i] = i;
      }
      std::random_device rd;
      std::mt19937 g(rd());
      std::shuffle(order_ptr, order_ptr + num_nodes, g);
      break;
    }
    case UMPolicy::kPreSample: {
      LOG(INFO) << "sort um dataset by PreSample";
      auto sampler = cuda::UMPreSampler(num_nodes, _num_step);
      sampler.DoPreSample();
      order = sampler.GetRankNode();
      break;
    }
    // ...
    default:
      order = Tensor::EmptyNoScale(DataType::kI32, {num_nodes}, CPU(), "");
      auto order_ptr = static_cast<IdType*>(order->MutableData());
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
      for(IdType i = 0; i < num_nodes; i++) {
        order_ptr[i] = i;
      }
      break;
    }
    if(RunConfig::unified_memory_policy != UMPolicy::kDefault) {
      SortUMDatasetBy(static_cast<const IdType*>(order->Data()));
    }
    LOG(INFO) << "sort um dataset " << sort_um_tm.Passed() << "secs";
  }

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
    case kArch7:
      RunArch7LoopsOnce();
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
    case kArch7:
      CHECK_EQ(_sampler_ctx, _trainer_ctx);
      CHECK(!RunConfig::UseGPUCache());
      break;
    default:
      CHECK(0);
  }
}

std::unordered_map<std::string, Context> GPUEngine::GetGraphFileCtx() {
  std::unordered_map<std::string, Context> ret;

  auto sampler_ctx = _sampler_ctx;
  if(RunConfig::unified_memory) {
    sampler_ctx.device_type = DeviceType::kGPU_UM;
  }

  ret[Constant::kIndptrFile] = sampler_ctx;
  ret[Constant::kIndicesFile] = sampler_ctx;
  ret[Constant::kTrainSetFile] = CPU();
  ret[Constant::kTestSetFile] = CPU();
  ret[Constant::kValidSetFile] = CPU();
  ret[Constant::kProbTableFile] = sampler_ctx;
  ret[Constant::kAliasTableFile] = sampler_ctx;
  ret[Constant::kInDegreeFile] = MMAP();
  ret[Constant::kOutDegreeFile] = MMAP();
  ret[Constant::kCacheByDegreeFile] = MMAP();
  ret[Constant::kCacheByHeuristicFile] = MMAP();
  ret[Constant::kCacheByDegreeHopFile] = MMAP();
  ret[Constant::kCacheByFakeOptimalFile] = MMAP();
  ret[Constant::kCacheByRandomFile] = MMAP();

  switch (RunConfig::run_arch) {
    case kArch1:
      ret[Constant::kFeatFile] = sampler_ctx;
      ret[Constant::kLabelFile] = sampler_ctx;
      break;
    case kArch2:
    case kArch3:
    case kArch7:
      ret[Constant::kFeatFile] = MMAP();
      ret[Constant::kLabelFile] = MMAP();
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

void GPUEngine::SortUMDatasetBy(const IdType* order) {
  size_t num_nodes = _dataset->indptr->Shape()[0] - 1;
  size_t indptr_nbytes =
    _dataset->indptr->Shape()[0] * GetDataTypeBytes(_dataset->indptr->Type());
  size_t indices_nbytes =
    _dataset->indices->Shape()[0] * GetDataTypeBytes(_dataset->indices->Type());

  IdType* nodeIdold2new = static_cast<IdType*>(Device::Get(CPU())->AllocWorkspace(
    CPU(), num_nodes * GetDataTypeBytes(DataType::kI32), Constant::kAllocNoScale));
  IdType* tmp_indptr = static_cast<IdType*>(Device::Get(CPU())->AllocWorkspace(
    CPU(), indptr_nbytes, Constant::kAllocNoScale));
  IdType* tmp_indices = static_cast<IdType*>(Device::Get(CPU())->AllocWorkspace(
    CPU(), indices_nbytes, Constant::kAllocNoScale));
  IdType* new_indptr = static_cast<IdType*>(Device::Get(CPU())->AllocWorkspace(
    CPU(), indptr_nbytes, Constant::kAllocNoScale));
  IdType* new_indices = static_cast<IdType*>(Device::Get(CPU())->AllocWorkspace(
    CPU(), indices_nbytes, Constant::kAllocNoScale));

  Device::Get(_dataset->indptr->Ctx().device_type == DeviceType::kMMAP ?
    CPU() : _dataset->indptr->Ctx())->CopyDataFromTo(
    _dataset->indptr->Data(), 0, tmp_indptr, 0, indptr_nbytes,
    _dataset->indptr->Ctx(), CPU());
  Device::Get(_dataset->indices->Ctx().device_type == DeviceType::kMMAP ?
    CPU() : _dataset->indices->Ctx())->CopyDataFromTo(
    _dataset->indices->Data(), 0, tmp_indices, 0, indices_nbytes,
    _dataset->indices->Ctx(), CPU());

#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for(IdType i = 0; i < num_nodes; i++) {
    nodeIdold2new[order[i]] = i;
  }

  new_indptr[0] = 0;
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for(IdType i = 1; i <= num_nodes; i++) {
    IdType v = order[i-1];
    CHECK(v >= 0 && v < num_nodes);
    new_indptr[i] = tmp_indptr[v+1] - tmp_indptr[v];
  }
  __gnu_parallel::partial_sum(
    new_indptr, new_indptr + _dataset->indptr->Shape()[0], new_indptr, std::plus<IdType>());
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for(IdType i = 0; i < num_nodes; i++) {
    IdType v = order[i];
    IdType old_off = tmp_indptr[v];
    IdType new_off = new_indptr[i];
    size_t edge_len = new_indptr[i+1] - new_indptr[i];
    CHECK(edge_len == tmp_indptr[v+1] - tmp_indptr[v]);
    for(IdType j = 0; j < edge_len; j++) {
      IdType u = tmp_indices[old_off + j];
      new_indices[new_off + j] = nodeIdold2new[u];
    }
  }

#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for(IdType i = 0; i < num_nodes; i++) {
    IdType v = nodeIdold2new[i];
    IdType old_off = tmp_indptr[i];
    IdType new_off = new_indptr[v];
    size_t edge_len = tmp_indptr[i+1] - tmp_indptr[i];
    for(IdType j = 0; j < edge_len; j++) {
      IdType u = new_indices[new_off + j];
      CHECK(order[u] == tmp_indices[old_off + j]);
    }
  }

  auto sort_edge_values = [&](TensorPtr &values) -> void {
    if(values == nullptr || values->Data() == nullptr)
      return;
    auto tmp_values_ts = Tensor::CopyTo(values, CPU());
    auto new_values_ts = Tensor::EmptyNoScale(
        values->Type(), values->Shape(), CPU(), values->Name());
    CHECK(tmp_values_ts->NumBytes() % tmp_values_ts->Shape()[0] == 0);
    auto per_edge_nbytes = (tmp_values_ts->NumBytes() / tmp_values_ts->Shape()[0]);
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
    for(IdType i = 0; i < num_nodes; i++) {
      IdType v = order[i];
      IdType old_off = tmp_indptr[v];
      IdType new_off = new_indptr[i];
      size_t edge_len = (new_indptr[i+1] - new_indptr[i]);
      Device::Get(new_values_ts->Ctx())->CopyDataFromTo(tmp_values_ts->Data(), old_off * per_edge_nbytes,
          new_values_ts->MutableData(), new_off * per_edge_nbytes,
          edge_len * per_edge_nbytes,
          tmp_values_ts->Ctx(), new_values_ts->Ctx());
    }
    if (values->Ctx().device_type == DeviceType::kMMAP) {
      values = new_values_ts;
    } else {
      Device::Get(values->Ctx())->CopyDataFromTo(new_values_ts->Data(), 0,
          values->MutableData(), 0,
          values->NumBytes(),
          new_values_ts->Ctx(), values->Ctx());
    }
  };
  sort_edge_values(_dataset->prob_table);
  sort_edge_values(_dataset->alias_table);
  sort_edge_values(_dataset->prob_prefix_table);

  auto sort_node_values = [&](TensorPtr &values) -> void {
    if(values == nullptr || values->Data() == nullptr)
      return;
    auto tmp_values_ts = Tensor::CopyTo(values, CPU());
    auto new_values_ts = Tensor::EmptyNoScale(
      values->Type(), values->Shape(), CPU(), values->Name());
    CHECK(tmp_values_ts->NumBytes() % tmp_values_ts->Shape()[0] == 0);
    auto per_node_nbytes = tmp_values_ts->NumBytes() / tmp_values_ts->Shape()[0];
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
    for(IdType i = 0; i < tmp_values_ts->Shape()[0]; i++) {
      size_t src = i;
      size_t dst = nodeIdold2new[i];
      memcpy(
        static_cast<char*>(new_values_ts->MutableData()) + dst * per_node_nbytes,
        static_cast<char*>(tmp_values_ts->MutableData()) + src * per_node_nbytes,
        per_node_nbytes
      );
    }
    if(values->Ctx().device_type == DeviceType::kMMAP) {
      values = new_values_ts;
    } else {
      Device::Get(values->Ctx())->CopyDataFromTo(
        new_values_ts->Data(), 0, values->MutableData(), 0, values->NumBytes(),
        CPU(), values->Ctx());
    }
  };
  sort_node_values(_dataset->in_degrees);
  sort_node_values(_dataset->out_degrees);
  if(!RunConfig::option_empty_feat)
    sort_node_values(_dataset->feat);
  sort_node_values(_dataset->label);

  auto sort_nodes = [&](TensorPtr &nodes) -> void {
    if(nodes == nullptr || nodes->Data() == nullptr)
      return;
    auto tmp_nodes_ts = Tensor::CopyTo(nodes, CPU());
    auto tmp_nodes = static_cast<IdType*>(tmp_nodes_ts->MutableData());
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
    for(IdType i = 0; i < nodes->Shape()[0]; i++) {
      CHECK(tmp_nodes[i] >= 0 && tmp_nodes[i] < num_nodes);
      IdType v = tmp_nodes[i];
      IdType u = nodeIdold2new[v];
      CHECK(tmp_indptr[v+1] - tmp_indptr[v] == new_indptr[u+1] - new_indptr[u]);
      tmp_nodes[i] = nodeIdold2new[tmp_nodes[i]];
    }
    if(nodes->Ctx().device_type == DeviceType::kMMAP) {
      nodes = tmp_nodes_ts;
    } else {
      Device::Get(nodes->Ctx())->CopyDataFromTo(
        tmp_nodes, 0, nodes->MutableData(), 0, nodes->NumBytes(),
        CPU(), nodes->Ctx());
    }
  };
  sort_nodes(_dataset->ranking_nodes);
  sort_nodes(_dataset->train_set);
  sort_nodes(_dataset->valid_set);
  sort_nodes(_dataset->test_set);

  if(_dataset->indptr->Ctx().device_type == DeviceType::kMMAP) {
    _dataset->indptr = Tensor::EmptyNoScale(
      _dataset->indptr->Type(), _dataset->indptr->Shape(), CPU(), "dataset.indptr");
  }
  if(_dataset->indices->Ctx().device_type == DeviceType::kMMAP) {
    _dataset->indices = Tensor::EmptyNoScale(
      _dataset->indices->Type(), _dataset->indices->Shape(), CPU(), "dataset.indices");
  }
  Device::Get(_dataset->indptr->Ctx())->CopyDataFromTo(
    new_indptr, 0, _dataset->indptr->MutableData(), 0, indptr_nbytes,
    CPU(), _dataset->indptr->Ctx());
  Device::Get(_dataset->indices->Ctx())->CopyDataFromTo(
    new_indices, 0, _dataset->indices->MutableData(), 0, indices_nbytes,
    CPU(), _dataset->indices->Ctx());

  // free tensor
  for(auto data : {nodeIdold2new, tmp_indptr, tmp_indices, new_indptr, new_indices}) {
    Device::Get(CPU())->FreeWorkspace(CPU(), data);
  }
}


}  // namespace cuda
}  // namespace common
}  // namespace samgraph
