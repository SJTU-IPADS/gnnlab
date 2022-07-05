#include <unordered_map>
#include <algorithm>
#include <memory>
#include "gtest/gtest.h"

#include "common/operation.h"
#include "common/common.h"
#include "common/run_config.h"
#include "common/dist/dist_engine.h"
#include "common/dist/dist_shuffler.h"

using namespace samgraph;
using namespace samgraph::common;

class Arch9Test : public ::testing::Test {
public:
protected:
  void SetUp() override {
    std::unordered_map<std::string, std::string> config = {
      // {"dataset_path",        "/graph-learning/samgraph/reddit/"},
      {"dataset_path",        "/graph-learning/samgraph/papers100M/"},

      {"_arch",               std::to_string(static_cast<int>(RunArch::kArch9))},
      {"_sample_type",        std::to_string(static_cast<int>(SampleType::kKHop0))},
      {"batch_size",          std::to_string(8000)},
      {"num_epoch",           std::to_string(2)},
      {"_cache_policy",       std::to_string(static_cast<int>(kCacheByDegree))},
      {"cache_percentage",    std::to_string(0)},
      {"max_sampling_jobs",   std::to_string(10)},
      {"max_copying_jobs",    std::to_string(1)},
      {"omp_thread_num",      std::to_string(40)},
      {"num_layer",           std::to_string(3)},
      {"num_hidden",          std::to_string(256)},
      {"lr",                  std::to_string(0.003)},
      {"dropout" ,            std::to_string(0.5)},

      {"num_fanout",          std::to_string(3)},
      {"fanout",              "5 10 15"},

      {"barriered_epoch",     std::to_string(0)},

      // arch9
      {"num_train_worker",   std::to_string(1)},
      {"num_sample_worker",  std::to_string(2)},
      {"unified_memory",     "True"},
      {"unified_memory_ctx",  "cuda:1 cuda:2"},
    };
    samgraph::common::samgraph_config_from_map(config);
    samgraph::common::samgraph_init();
  }

  void TearDown() override {
    samgraph::common::samgraph_shutdown();
  }
};

TEST_F(Arch9Test, task_train_set) {
  samgraph::common::samgraph_um_sample_init(RunConfig::num_sample_worker);
  
  auto origin_train_set = Tensor::CopyTo(Engine::Get()->GetGraphDataset()->train_set, CPU());
  auto origin_train_set_ptr = static_cast<IdType*>(origin_train_set->MutableData());
  std::sort(origin_train_set_ptr, origin_train_set_ptr + origin_train_set->Shape()[0]);

  auto& samplers = dist::DistEngine::Get()->GetUMSamplers();

  int local_num_step = 0;
  for (int i = 0; i < RunConfig::num_sample_worker; i++) {
    auto shuffler = dynamic_cast<dist::DistShuffler*>(samplers[i]->GetShuffler());
    local_num_step = std::max(local_num_step, (int)shuffler->NumLocalStep());
  }
  int num_step = Engine::Get()->NumStep();

  auto queue = dynamic_cast<MessageTaskQueue*>(dist::DistEngine::Get()->GetTaskQueue(cuda::kDataCopy));
  for (int e = 0; e < RunConfig::num_epoch; e++) {
    for (int s = 0; s < local_num_step; s++) {
      samgraph::common::samgraph_sample_once();
    }

    // check trainset size
    std::vector<TensorPtr> train_set_ts;
    for (int s = 0; s < num_step; s++) {
      auto task = queue->Recv();
      train_set_ts.push_back(Tensor::CopyTo(task->output_nodes, CPU()));
    }
    int train_set_num = 0;
    for (auto &ts : train_set_ts) {
      train_set_num += ts->Shape()[0];
    }
    EXPECT_EQ(train_set_num, (int)origin_train_set->Shape()[0]);

    // train set should be equal
    auto cur_train_set_ptr = std::make_unique<IdType[]>(train_set_num);
    for (IdType i = 0, off = 0; i < train_set_ts.size(); i++) {
      auto src = static_cast<const IdType*>(train_set_ts[i]->Data());
      std::memcpy(cur_train_set_ptr.get() + off, src, train_set_ts[i]->NumBytes());
      off += train_set_ts[i]->Shape()[0];
    }
    std::sort(cur_train_set_ptr.get(), cur_train_set_ptr.get() + train_set_num);
    for (int i = 0; i < train_set_num; i++) {
      EXPECT_EQ(cur_train_set_ptr[i], origin_train_set_ptr[i]);
    }
  }
}