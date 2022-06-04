#include <chrono>
#include <numeric>
#include <semaphore.h>

#include "../device.h"
#include "../function.h"
#include "../logging.h"
#include "../profiler.h"
#include "../run_config.h"
#include "../timer.h"
#include "../common.h"

#include "dist_loops.h"
#include "dist_shuffler.h"
#include "../cuda/cuda_function.h"
#include "../cuda/cuda_hashtable.h"
#include "../cuda/cuda_loops.h"

/**
 *  multi-sampler(thread) --> trainer (based on arch5)
 *  
 */

namespace samgraph {
namespace common {
namespace dist {

namespace {

bool RunSampleSubLoopOnce() {
  auto tid = std::this_thread::get_id();
  DistUMSampler* sampler = DistEngine::Get()->GetUMSamplerByTid(tid);
  CHECK(sampler != nullptr);
  LOG(DEBUG) << "sampler(" << sampler->Ctx() << ") "
             << "worker(" << std::hex << tid  << ")" << " start sample once";
  CHECK(sampler != nullptr);
  auto graph_pool = sampler->GetGraphPool();
  if (graph_pool->Full()) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    return true;
  }

  auto next_op = cuda::kDataCopy;
  auto next_q = dynamic_cast<MessageTaskQueue*>(DistEngine::Get()->GetTaskQueue(next_op));
  if (next_q->Full()) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    return true;
  }
  double shuffle_time, sample_time, get_miss_cache_index_time, send_time;

  Timer t_total;
  Timer t0;
  auto task = DoShuffle();
  if (task == nullptr) {
    LOG(FATAL) << "null task from DoShuffle";
  }
  shuffle_time = t0.Passed();

  LOG(DEBUG) << "sampler(" << sampler->Ctx() << ") get task, ready to sample"
             << " task_key=" << task->key
             << " epoch=" << sampler->GetShuffler()->Epoch()
             << " global_step=" << sampler->GetShuffler()->Step();

  Timer t1;
  DoGPUSample(task);
  sample_time = t1.Passed();

  Timer t2;
  DoGetCacheMissIndex(task);
  get_miss_cache_index_time = t2.Passed();

  LOG(TRACE) << "sampler(" << sampler->Ctx() << ") sample & cache miss index done";

  Timer t3;
  next_q->Send(task);
  send_time = t3.Passed();

  // epoch profile 
  LOG(DEBUG) << "sampler(" << sampler->Ctx() << ") task_key=" << task->key
             << " sample time " << sample_time;
  Profiler::Get().LogEpochAdd(task->key, kLogEpochSampleTime,
                              shuffle_time + sample_time);
  Profiler::Get().LogEpochAdd(task->key, KLogEpochSampleGetCacheMissIndexTime,
                              get_miss_cache_index_time);
  Profiler::Get().LogEpochAdd(task->key, kLogEpochSampleSendTime, send_time);
  Profiler::Get().LogEpochAdd(
      task->key, kLogEpochSampleTotalTime,
      shuffle_time + sample_time + get_miss_cache_index_time + send_time);
  // step profile
  Profiler::Get().LogStep(task->key, kLogL1SampleTime,
                          shuffle_time + sample_time);
  Profiler::Get().LogStep(task->key, kLogL1SendTime,
                          send_time);
  Profiler::Get().LogStep(task->key, kLogL2ShuffleTime, shuffle_time);
  Profiler::Get().LogStep(task->key, kLogL1SamplerId, sampler->Ctx().device_id);

  LOG(DEBUG) << "sampler(" << sampler->Ctx() << ") sample once done "
             << "task_key=" << task->key;
  return true;
} 


bool RunSample(sem_t* sem) {
  sem_wait(sem);
  auto tid = std::this_thread::get_id();
  auto sampler = DistEngine::Get()->GetUMSamplerByTid(tid);
  auto ctx = sampler->Ctx();
  CHECK(sampler != nullptr);
  while (1) {
    sem_wait(sem);
    RunSampleSubLoopOnce();

    auto shuffer = dynamic_cast<DistShuffler*>(sampler->GetShuffler());
    auto epoch = shuffer->Epoch();
    auto step = shuffer->Step();
    LOG(DEBUG) << "sampler(" << ctx << ") sampling done, "
               << " epoch=" << epoch
               << " global step=" << step;
    if (epoch + 1 == sampler->GetShuffler()->NumEpoch() &&  shuffer->IsLastBatch()) {
      break;
    }
  }
  LOG(INFO) << "sampler(" << ctx << ") "
            << "worker(" << std::hex << tid << ") task done, exiting...";
  return true;
}

bool RunDataCopySubLoopOnce() {
  auto graph_pool = DistEngine::Get()->GetGraphPool();
  while (graph_pool->Full()) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    // return true;
  }

  auto this_op = cuda::kDataCopy;
  auto q = dynamic_cast<MessageTaskQueue*>(DistEngine::Get()->GetTaskQueue(this_op));
  Timer t4;
  auto task = q->Recv();
  double recv_time = t4.Passed();

  if (task) {
    Timer t0;
    DoGraphCopy(task);
    double graph_copy_time = t0.Passed();

    Timer t2;
    DoCPUFeatureExtract(task);
    double extract_time = t2.Passed();

    Timer t3;
    DoFeatureCopy(task);
    double feat_copy_time = t3.Passed();

    LOG(DEBUG) << "Submit: process task with key " << task->key;
    graph_pool->Submit(task->key, task);

    Profiler::Get().LogStep(
        task->key, kLogL1CopyTime,
        recv_time + graph_copy_time + extract_time + feat_copy_time);
    Profiler::Get().LogStep(task->key, kLogL1RecvTime, recv_time);
    Profiler::Get().LogStep(task->key, kLogL2GraphCopyTime, graph_copy_time);
    Profiler::Get().LogStep(task->key, kLogL2ExtractTime, extract_time);
    Profiler::Get().LogStep(task->key, kLogL2FeatCopyTime, feat_copy_time);
    Profiler::Get().LogEpochAdd(
        task->key, kLogEpochCopyTime,
        recv_time + graph_copy_time + extract_time + feat_copy_time);
  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }

  return true;
}

bool RunCacheDataCopySubLoopOnce() {
  auto graph_pool = DistEngine::Get()->GetGraphPool();
  auto dist_type  = DistEngine::Get()->GetDistType();
  while (graph_pool->Full()) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    // return true;
  }

  auto this_op = cuda::kDataCopy;
  auto q = dynamic_cast<MessageTaskQueue*>(DistEngine::Get()->GetTaskQueue(this_op));
  // receive the task data from sample process
  Timer t4;
  auto task = q->Recv();
  double recv_time = t4.Passed();

  if (task) {
    Timer t0;
    DoGraphCopy(task);
    double graph_copy_time = t0.Passed();

    Timer t2;
    switch(dist_type) {
      case (DistType::Extract): {
            DoCacheFeatureCopy(task);
            break;
      }
      case (DistType::Switch): {
            DoSwitchCacheFeatureCopy(task);
            break;
      }
      default:
            CHECK(0);
    }
    DoCPULabelExtractAndCopy(task);
    double cache_feat_copy_time = t2.Passed();

    LOG(DEBUG) << "Submit with cache: process task with key " << task->key;
    graph_pool->Submit(task->key, task);

    Profiler::Get().LogStep(task->key, kLogL1CopyTime,
                            recv_time + graph_copy_time + cache_feat_copy_time);
    Profiler::Get().LogStep(task->key, kLogL1RecvTime, recv_time);
    Profiler::Get().LogStep(task->key, kLogL2GraphCopyTime, graph_copy_time);
    Profiler::Get().LogStep(task->key, kLogL2CacheCopyTime,
                            cache_feat_copy_time);
    Profiler::Get().LogEpochAdd(
        task->key, kLogEpochCopyTime,
        recv_time + graph_copy_time + cache_feat_copy_time);
  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }
  return true;
}

void DataCopySubLoop(int count) {
  LoopOnceFunction func;
  if (!RunConfig::UseGPUCache()) {
    func = RunDataCopySubLoopOnce;
  } else {
    func = RunCacheDataCopySubLoopOnce;
  }

  while ((count--) && !DistEngine::Get()->ShouldShutdown() && func());

  DistEngine::Get()->ReportThreadFinish();
}

} // namespace

void RunArch9LoopsOnce(DistType dist_type) {
  if (dist_type == DistType::Sample) {
    static int epoch_run_loops_cnt = 0;
    static int step_run_loops_cnt = 0;
    LOG(TRACE) << "RunArch9LoopsOnce with Sample, "
               << "epoch=" << epoch_run_loops_cnt << " "
               << "step="  << step_run_loops_cnt;

    int num_epoch = DistEngine::Get()->NumEpoch();
    int num_step = DistEngine::Get()->NumStep();
    int local_sampler_step = num_step / RunConfig::num_sample_worker;
    int last_sampler_step = num_step - (local_sampler_step * (RunConfig::num_sample_worker - 1));
    CHECK(local_sampler_step <= last_sampler_step);

    auto um_samplers = DistEngine::Get()->GetUMSamplers();
    if (epoch_run_loops_cnt == 0 && step_run_loops_cnt == 0) {
      for (int i = 0; i < um_samplers.size(); i++) {
        auto sampler = um_samplers[i];
        if (epoch_run_loops_cnt == 0 && step_run_loops_cnt == 0 &&
          sampler->WorkerId() == static_cast<std::thread::id>(0)
        ) {
          sampler->CreateWorker(RunSample, sampler->WorkerSem());
          LOG(INFO) << "sampler(" << sampler->Ctx() << ") " 
                    << "create sample worker(0x" << std::hex << sampler->WorkerId() << ")";
        }
      }
      for (int i = 0; i < um_samplers.size(); i++) {
        um_samplers[i]->ReleaseSem();
      }
    }

    // RunSampleSubLoopOnce();
    for (int i = 0; i < um_samplers.size(); i++) { 
      auto sampler = um_samplers[i];
      if (i + 1 == um_samplers.size()) {
        LOG(DEBUG) << "sampler(" << sampler->Ctx() <<  ") issue sampling task, "
                  << "epoch=" << epoch_run_loops_cnt
                  << " local step=" << step_run_loops_cnt;
        sampler->ReleaseSem();
      } else {
        if (step_run_loops_cnt < local_sampler_step) {
          LOG(DEBUG) << "sampler(" << sampler->Ctx() <<  ") issue sampling task, "
                    << "epoch=" << epoch_run_loops_cnt
                    << " local step=" << step_run_loops_cnt;
          sampler->ReleaseSem();
        }
      }
    }
    step_run_loops_cnt++;
    if (step_run_loops_cnt == last_sampler_step) {
      step_run_loops_cnt = 0;
      epoch_run_loops_cnt++;
    }
  }
  else if (dist_type == DistType::Extract || dist_type == DistType::Switch) {
    if (!RunConfig::UseGPUCache()) {
      LOG(DEBUG) << "RunArch9LoopsOnce with Extract no Cache!";
      RunDataCopySubLoopOnce();
    } else {
      LOG(DEBUG) << "RunArch9LoopsOnce with Extract Cache!";
      RunCacheDataCopySubLoopOnce();
    }
  } else {
    LOG(FATAL) << "dist type is illegal!";
  }
} 

ExtractFunction GetArch8Loops() {
  return DataCopySubLoop;
}


} // namespace dist
} // namespace common
} // namespace samgraph
