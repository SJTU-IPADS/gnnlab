#include <unordered_set>
#include <functional>

#include "gtest/gtest.h"
#include "common/common.h"
#include "common/logging.h"
#include "common/device.h"
#include "common/run_config.h"
#include "common/cuda/cuda_hashtable.h"

namespace {
using namespace samgraph;
using namespace samgraph::common;

using HashTablePtr = std::shared_ptr<cuda::OrderedHashTable>;
using uset = std::unordered_set<IdType>;
using vec = std::vector<IdType>;

// reuse the test class to run multiple test w/ or w/o reseting the table
enum class ETearDown {
  kDoReset,
  kMakeNewTable,
};

class OrderedHashTableTest: public ::testing::TestWithParam<ETearDown> { 
public:
  static void SetUpTestSuite() {
    ctx = Context{kGPU, 0};
    cuda_device = Device::Get(ctx);
    stream = cuda_device->CreateStream(ctx);
    hash_table = std::make_shared<cuda::OrderedHashTable>(1024, ctx, stream);
  }
  static void TearDownTestSuite() {
    hash_table = nullptr;
    cuda_device->FreeStream(ctx, stream);
  }

  void TearDown() override {
    // reset the hashtable and reuse it, or make a new one
    if (GetParam() == ETearDown::kDoReset) {
      hash_table->Reset(stream);
    } else {
      hash_table = std::make_shared<cuda::OrderedHashTable>(1024, ctx, stream);
    }
  }

  template<typename T>
  static T* _CopyToDevice(const T* const ptr, size_t num_item) {
    T* ret;
    CUDA_CALL(cudaMalloc(&ret, num_item * sizeof(T)));
    _CopyFromTo(ptr, ret, num_item);
    return ret;
  }
  template<typename T>
  static void _CopyFromTo(const T* const from, T* to, size_t num_item) {
    // let cuda auto infer src & dst
    CUDA_CALL(cudaMemcpy(to, from,num_item * sizeof(T), cudaMemcpyDefault));
  }

  template<typename Iter>
  static void EXPECT_Contain(uset & set, Iter begin, Iter end) {
    for (Iter i = begin; i != end; i++) {
      EXPECT_EQ(set.count(*i), 1);
    }
  }

  static Context ctx;
  static Device* cuda_device;
  static StreamHandle stream;
  static HashTablePtr hash_table;
};

Context OrderedHashTableTest::ctx;
Device* OrderedHashTableTest::cuda_device = nullptr;
StreamHandle OrderedHashTableTest::stream = nullptr;
HashTablePtr OrderedHashTableTest::hash_table = nullptr;
} // namespace

TEST_P(OrderedHashTableTest, DupRevised_Ref) {
  auto FillAndGetMethod = [](vec input, vec& output_uniq) {
    IdType *d_data = _CopyToDevice(input.data(), input.size());
    hash_table->FillWithDupRevised(d_data, input.size(), stream);
    cuda_device->StreamSync(ctx, stream);
    CUDA_CALL(cudaFree(d_data));

    const IdType *data_unique;
    IdType num_unique;
    hash_table->RefUnique(data_unique, &num_unique);
    output_uniq.resize(num_unique);
    _CopyFromTo(data_unique, output_uniq.data(), num_unique);
  };
  /* ===================== First Insert ===================== */
  vec h_data = {1, 2, 3, 4, 5, 2, 5, 1, 10, 233}, output_uniq;
  FillAndGetMethod(h_data, output_uniq);

  // make sure it's really unique and correct
  auto d_set = uset(output_uniq.begin(), output_uniq.end());
  uset h_set(h_data.begin(), h_data.end());
  EXPECT_EQ(d_set, h_set);

  /* ===================== Second Insert ===================== */
  h_data = {1, 2, 3, 6, 9, 2, 8, 1, 3, 7, 1023};
  FillAndGetMethod(h_data, output_uniq);

  // validate the prefix is maintained
  EXPECT_EQ(uset(output_uniq.begin(), output_uniq.begin() + h_set.size()),
            h_set);
  // make sure it's unique
  d_set = uset(output_uniq.begin(), output_uniq.end());
  EXPECT_EQ(d_set.size(), output_uniq.size());
  h_set.insert(h_data.begin(), h_data.end());
  EXPECT_EQ(d_set, h_set);

  /* ===================== Third Insert ===================== */
  // simply concat first 2 array
  h_data = {1, 2, 3, 4, 5, 2, 5, 1, 10, 233,
            1, 2, 3, 6, 9, 2, 8, 1, 3, 7, 1023};
  FillAndGetMethod(h_data, output_uniq);

  d_set = uset(output_uniq.begin(), output_uniq.end());
  h_set = uset(h_data.begin(), h_data.end());
  EXPECT_EQ(d_set, h_set);
}

TEST_P(OrderedHashTableTest, MixedFillMethod) {
  auto FillAndGet_1 = [](vec input, vec&output_uniq) {
    IdType *data_unique, num_unique;
    CUDA_CALL(cudaMalloc(&data_unique, 1024 * sizeof(IdType)));
    IdType *d_data = _CopyToDevice(input.data(), input.size());

    hash_table->FillWithDuplicates(d_data, input.size(), data_unique, &num_unique, stream);
    cuda_device->StreamSync(ctx, stream);

    output_uniq.resize(num_unique);
    _CopyFromTo(data_unique, output_uniq.data(), num_unique);
    CUDA_CALL(cudaFree(d_data));
    CUDA_CALL(cudaFree(data_unique));
  };
  auto FillAndGet_2 = [](vec input, vec& output_uniq) {
    IdType *data_unique, num_unique;
    CUDA_CALL(cudaMalloc(&data_unique, 1024 * sizeof(IdType)));
    IdType *d_data = _CopyToDevice(input.data(), input.size());

    hash_table->FillWithDupMutable(d_data, input.size(), stream);
    hash_table->CopyUnique(data_unique, stream);
    cuda_device->StreamSync(ctx, stream);
    num_unique = hash_table->NumItems();

    output_uniq.resize(num_unique);
    _CopyFromTo(data_unique, output_uniq.data(), num_unique);

    CUDA_CALL(cudaFree(d_data));
    CUDA_CALL(cudaFree(data_unique));
  };

  vec h_data = { // unique items count: 13
    1, 2, 3, 4, 5, 2, 5, 1, 10, 256,
    6, 9, 2, 13, 5, 64, 512, 1021};
  uset h_set(h_data.begin(), h_data.end());
  EXPECT_EQ(h_set.size(), 13);
  vec output_uniq;

  // FillWithDuplicate
  FillAndGet_1(h_data, output_uniq);

  auto d_set = uset(output_uniq.begin(), output_uniq.end());
  EXPECT_EQ(d_set, h_set);

  h_data.resize(512);
  for (int i = 1; i <= 512; ++i) {
    h_data[i - 1] = i;
  }

  // FillWithDupMutable, CopyUnique
  FillAndGet_2(h_data, output_uniq);

  EXPECT_EQ(output_uniq.size(), 513);
  // check prefix of unique array
  EXPECT_EQ(uset(output_uniq.begin(), output_uniq.begin() + h_set.size()),
            h_set);
  // check entire unique array
  d_set = uset(output_uniq.begin(), output_uniq.end());
  EXPECT_EQ(d_set.size(), output_uniq.size());
  h_set.insert(h_data.begin(), h_data.end());
  EXPECT_EQ(d_set, h_set);
}

INSTANTIATE_TEST_SUITE_P(ResetTable, OrderedHashTableTest, 
    ::testing::Values(ETearDown::kDoReset, ETearDown::kMakeNewTable));
