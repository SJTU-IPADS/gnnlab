#include "../profiler.h"
#include "../common.h"
#include "../constant.h"
#include "../logging.h"
#include "../device.h"
#include "dist_loops.h"
#include "dist_engine.h"
#include "pre_sampler.h"
#include "../cuda/cuda_utils.h"
#include <cstring>
#include "../timer.h"
#include "../cuda/cub_sort_wrapper.h"

namespace samgraph {
namespace common {
namespace dist {

namespace {

#define SAM_1D_GRID_INIT(num_input) \
  const size_t num_tiles = RoundUpDiv((size_t)num_input, Constant::kCudaTileSize); \
  const dim3 grid(num_tiles);                                              \
  const dim3 block(Constant::kCudaBlockSize);                              \

#define SAM_1D_GRID_FOR(loop_iter, num_input) \
  assert(BLOCK_SIZE == blockDim.x);                       \
  const size_t block_start = TILE_SIZE * blockIdx.x;      \
  const size_t block_end = min(TILE_SIZE * (blockIdx.x + 1), num_input);  \
  for (size_t loop_iter = threadIdx.x + block_start; loop_iter < block_end; loop_iter += BLOCK_SIZE) \

template <size_t BLOCK_SIZE=Constant::kCudaBlockSize, size_t TILE_SIZE=Constant::kCudaTileSize>
__global__ void record_freq(Id64Type *freq_table, const IdType *input_nodes, size_t num_inputs) {
  SAM_1D_GRID_FOR(i, num_inputs) {
    auto freq_ptr = reinterpret_cast<IdType*>(&freq_table[input_nodes[i]]);
    *(freq_ptr+1) += 1;
  }
}


template <size_t BLOCK_SIZE=Constant::kCudaBlockSize, size_t TILE_SIZE=Constant::kCudaTileSize>
__global__ void get_high_32(IdType *high, const Id64Type *src, size_t num_inputs) {
  SAM_1D_GRID_FOR(i, num_inputs) {
    high[i] = static_cast<IdType>(src[i] >> 32);
  }
}

template <size_t BLOCK_SIZE=Constant::kCudaBlockSize, size_t TILE_SIZE=Constant::kCudaTileSize>
__global__ void get_low_32(IdType *low, const Id64Type *src, size_t num_inputs) {
  SAM_1D_GRID_FOR(i, num_inputs) {
    low[i] = static_cast<IdType>(src[i]);
  }
}
}

PreSampler* PreSampler::singleton = nullptr;
PreSampler::PreSampler(TensorPtr input, size_t batch_size, size_t num_nodes) {
  Timer t_init;
  auto stream = DistEngine::Get()->GetSampleStream();
  auto sampler_ctx = DistEngine::Get()->GetSamplerCtx();
  auto sampler_device = Device::Get(sampler_ctx);
  LOG(ERROR) << "Dist Presampler making shuffler...\n";
  _shuffler = new cuda::GPUShuffler(input, RunConfig::presample_epoch,
                          batch_size, false);
  LOG(ERROR) << "Dist Presampler making shuffler...Done\n";
  _num_nodes = num_nodes;
  _num_step = _shuffler->NumStep();
  freq_table = Tensor::Empty(kI64, {_num_nodes}, sampler_ctx, "pre_sampler_freq");
  // freq_table = static_cast<Id64Type*>(Device::Get(CPU())->AllocDataSpace(CPU(), sizeof(Id64Type)*_num_nodes));
  cuda::ArrangeArray<Id64Type>(freq_table->Ptr<Id64Type>(), _num_nodes, 0, 1, stream);
  sampler_device->StreamSync(sampler_ctx, stream);
// #pragma omp parallel for num_threads(RunConfig::omp_thread_num)
//   for (size_t i = 0; i < _num_nodes; i++) {
//     auto nid_ptr = reinterpret_cast<IdType*>(&freq_table[i]);
//     *nid_ptr = i;
//     *(nid_ptr + 1) = 0;
//   }
  Profiler::Get().LogInit(kLogInitL3PresampleInit, t_init.Passed());
}

PreSampler::~PreSampler() {
  // Device::Get(CPU())->FreeDataSpace(CPU(), freq_table);
  delete _shuffler;
}

TaskPtr PreSampler::DoPreSampleShuffle() {
  auto s = _shuffler;
  auto stream = DistEngine::Get()->GetSampleStream();
  auto batch = s->GetBatch(stream);

  if (batch) {
    auto task = std::make_shared<Task>();
    task->output_nodes = batch;
    task->key = DistEngine::Get()->GetBatchKey(s->Epoch(), s->Step());
    LOG(DEBUG) << "DoShuffle: process task with key " << task->key;
    return task;
  } else {
    return nullptr;
  }
}

void PreSampler::DoPreSample(){
  auto sampler_ctx = DistEngine::Get()->GetSamplerCtx();
  auto sampler_device = Device::Get(sampler_ctx);
  auto cpu_device = Device::Get(CPU());
  auto stream = DistEngine::Get()->GetSampleStream();
  auto cu_stream = static_cast<cudaStream_t>(stream);
  for (int e = 0; e < RunConfig::presample_epoch; e++) {
    LOG(ERROR) << "Dist Presampler doing presample epoch " << e;
    for (size_t i = 0; i < _num_step; i++) {
      Timer t0;
      auto task = DoPreSampleShuffle();
      switch (RunConfig::cache_policy) {
        case kCollCache:
        case kCacheByPreSample:
          DoGPUSample(task);
          break;
        case kCacheByPreSampleStatic:
          LOG(FATAL) << "kCacheByPreSampleStatic is not implemented in DistEngine now!";
          /*
          DoGPUSampleAllNeighbour(task);
          break;
          */
        default:
          CHECK(0);
      }
      double sample_time = t0.Passed();
      size_t num_inputs = task->input_nodes->Shape()[0];
      Timer t2;
      {
        SAM_1D_GRID_INIT(num_inputs);
        record_freq<><<<grid, block, 0, cu_stream>>>(freq_table->Ptr<Id64Type>(), task->input_nodes->CPtr<IdType>(), num_inputs);
        sampler_device->StreamSync(sampler_ctx, stream);
      }
      double count_time = t2.Passed();
      Profiler::Get().LogInitAdd(kLogInitL3PresampleSample, sample_time);
      Profiler::Get().LogInitAdd(kLogInitL3PresampleCount, count_time);
    }
    LOG(ERROR) << "presample spend "
               << Profiler::Get().GetLogInitValue(kLogInitL3PresampleSample) << " on sample, "
               << Profiler::Get().GetLogInitValue(kLogInitL3PresampleCount) << " on count";
  }
  Timer ts;
  auto freq_table_sorted = Tensor::Empty(kI64, {_num_nodes}, sampler_ctx, "");
  cuda::CubSortKeyDescending(freq_table->CPtr<Id64Type>(), freq_table_sorted->Ptr<Id64Type>(), _num_nodes, sampler_ctx, 0, sizeof(Id64Type)*8, stream);
  freq_table = freq_table_sorted;
  double sort_time = ts.Passed();
  Profiler::Get().LogInit(kLogInitL3PresampleSort, sort_time);
  Timer t_reset;
  Profiler::Get().ResetStepEpoch();
  Profiler::Get().LogInit(kLogInitL3PresampleReset, t_reset.Passed());
}

TensorPtr PreSampler::GetFreq() {
  auto ranking_freq = Tensor::Empty(DataType::kI32, {_num_nodes}, CPU(), "");
  auto ranking_freq_ptr = static_cast<IdType*>(ranking_freq->MutableData());
  GetFreq(ranking_freq_ptr);
  return ranking_freq;
}

// please make sure ptr is accessible from gpu!
void PreSampler::GetFreq(IdType* ranking_freq_ptr) {
  auto sampler_ctx = DistEngine::Get()->GetSamplerCtx();
  auto sampler_device = Device::Get(sampler_ctx);
  auto stream = DistEngine::Get()->GetSampleStream();
  auto cu_stream = static_cast<cudaStream_t>(stream);
  SAM_1D_GRID_INIT(_num_nodes);
  get_high_32<><<<grid, block, 0, cu_stream>>>(ranking_freq_ptr, freq_table->CPtr<Id64Type>(), _num_nodes);
  sampler_device->StreamSync(sampler_ctx, stream);
}
TensorPtr PreSampler::GetRankNode() {
  auto ranking_nodes = Tensor::Empty(DataType::kI32, {_num_nodes}, CPU(), "");
  GetRankNode(ranking_nodes);
  return ranking_nodes;
}
void PreSampler::GetRankNode(TensorPtr& ranking_nodes) {
  auto ranking_nodes_ptr = ranking_nodes->Ptr<IdType>();
  GetRankNode(ranking_nodes_ptr);
}

void PreSampler::GetRankNode(IdType* ranking_nodes_ptr) {
  auto sampler_ctx = DistEngine::Get()->GetSamplerCtx();
  auto sampler_device = Device::Get(sampler_ctx);
  auto stream = DistEngine::Get()->GetSampleStream();
  auto cu_stream = static_cast<cudaStream_t>(stream);
  SAM_1D_GRID_INIT(_num_nodes);
  Timer t_prepare_rank;
  get_low_32<><<<grid, block, 0, cu_stream>>>(ranking_nodes_ptr, freq_table->CPtr<Id64Type>(), _num_nodes);
  sampler_device->StreamSync(sampler_ctx, stream);
  Profiler::Get().LogInit(kLogInitL3PresampleGetRank, t_prepare_rank.Passed());
}


}
}
}