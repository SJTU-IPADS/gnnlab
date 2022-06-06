#include <unordered_set>

#include "gtest/gtest.h"
#include "common/common.h"
#include "common/logging.h"
#include "common/device.h"
#include "common/run_config.h"
#include "common/cuda/cuda_hashtable.h"

using namespace samgraph;
using namespace samgraph::common;

using HashTablePtr = std::shared_ptr<cuda::OrderedHashTable>;

void _CheckResult(const std::vector<IdType> &input_data,
    const HashTablePtr hash_table, const size_t check_item) {

  size_t num_items = hash_table->NumItems();
  std::unordered_set<IdType> h_set(input_data.begin(), input_data.end());
  EXPECT_TRUE(h_set.size() <= num_items)
    << "size of h_set and device_hash_table: "
    << h_set.size() << " vs " << num_items;
  EXPECT_EQ(num_items, check_item);

  const IdType *d_n2o;
  IdType num_unique;
  hash_table->RefUnique(d_n2o, &num_unique);
  EXPECT_EQ(num_unique, check_item);

  std::vector<IdType> h_n2o(num_items);
  CUDA_CALL(cudaMemcpy(h_n2o.data(), d_n2o, num_items * sizeof(IdType),
        cudaMemcpyDeviceToHost));
  std::unordered_set<IdType> check_set(h_n2o.begin(), h_n2o.end());
  EXPECT_EQ(check_set.size(), h_n2o.size());

  for (auto i : input_data) {
    EXPECT_EQ(check_set.count(i), 1);
  }
}

void _TestOrderedHashTable() {
  auto ctx = Context{kGPU, 0};
  auto cuda_device = Device::Get(ctx);
  auto stream = cuda_device->CreateStream(ctx);
  auto hash_table = std::make_shared<cuda::OrderedHashTable>(
      1024, ctx, stream);
  hash_table->Reset(stream);

  { // check hash_table functions: NumItems, FillWithDupRevised, RefUnique

  auto lambda_FillWithDupRevised = [&](std::vector<IdType> &h_insert_data) {
    IdType *d_data;
    CUDA_CALL(cudaMalloc(&d_data, h_insert_data.size() * sizeof(IdType)));
    CUDA_CALL(cudaMemcpy(d_data, h_insert_data.data(),
        h_insert_data.size() * sizeof(IdType),
        cudaMemcpyHostToDevice));
    hash_table->FillWithDupRevised(d_data, h_insert_data.size(), stream);
    // stream sync
    cuda_device->StreamSync(ctx, stream);
    CUDA_CALL(cudaFree(d_data));
  };

  // insert data to hash_table
  std::vector<IdType> h_data = {
    1, 2, 3, 4, 5, 2, 5, 1, 10, 233};
  lambda_FillWithDupRevised(h_data);
  _CheckResult(h_data, hash_table, 7);

  std::vector<IdType> h_data2 = {
    1, 2, 3, 6, 9, 2, 8, 1, 3, 7, 1023};
  lambda_FillWithDupRevised(h_data2);
  _CheckResult(h_data2, hash_table, 12);

  std::vector<IdType> h_data3;
  h_data3.reserve(h_data.size() + h_data2.size());
  h_data3.insert(h_data3.end(), h_data.begin(), h_data.end());
  h_data3.insert(h_data3.end(), h_data2.begin(), h_data2.end());
  lambda_FillWithDupRevised(h_data3);
  _CheckResult(h_data3, hash_table, 12);

  } // check hash_table functions: NumItems, FillWithDupRevised, RefUnique

  hash_table->Reset(stream);
  cuda_device->StreamSync(ctx, stream);
  EXPECT_EQ(hash_table->NumItems(), 0);

  { // check functions:
    //   FillWithDuplicates, NumItems, FillWithDupMutable, CopyUnique

  // FillWithDuplicates
  std::vector<IdType> h_data = { // unique items count: 13
    1, 2, 3, 4, 5, 2, 5, 1, 10, 256,
    6, 9, 2, 13, 5, 64, 512, 1021};
  std::unordered_set<IdType> check_set = {
    1, 2, 3, 4, 5, 10, 256, 6, 9, 13, 64, 512, 1021};
  EXPECT_EQ(check_set.size(), 13);

  IdType *d_data, *data_unique;
  IdType num_unique;
  CUDA_CALL(cudaMalloc(&d_data, h_data.size() * sizeof(IdType)));
  CUDA_CALL(cudaMallocManaged(&data_unique,
        1024 * sizeof(IdType)));
  CUDA_CALL(cudaMemcpy(d_data, h_data.data(),
        h_data.size() * sizeof(IdType),
        cudaMemcpyHostToDevice));
  hash_table->FillWithDuplicates(d_data, h_data.size(),
        data_unique, &num_unique,
        stream);
  cuda_device->StreamSync(ctx, stream);

  EXPECT_EQ(num_unique, 13);
  for (IdType i = 0 ; i < num_unique; ++i) {
    EXPECT_EQ(check_set.count(data_unique[i]), 1);
  }
  CUDA_CALL(cudaFree(d_data));
  d_data = nullptr;

  // FillWithDupMutable, CopyUnique
  std::vector<IdType> h_data2(512);
  for (int i = 1; i <= 512; ++i) {
    h_data2[i - 1] = i;
  }

  CUDA_CALL(cudaMalloc(&d_data, h_data2.size() * sizeof(IdType)));
  CUDA_CALL(cudaMemcpy(d_data, h_data2.data(),
        h_data2.size() * sizeof(IdType),
        cudaMemcpyHostToDevice));
  hash_table->FillWithDupMutable(d_data, h_data2.size(), stream);
  hash_table->CopyUnique(data_unique, stream);
  cuda_device->StreamSync(ctx, stream);
  num_unique = hash_table->NumItems();
  EXPECT_EQ(num_unique, 513);
  for (int i = 0; i < 13; ++i) {
    EXPECT_EQ(check_set.count(data_unique[i]), 1);
  }
  check_set.insert(h_data2.begin(), h_data2.end());
  EXPECT_EQ(check_set.size(), 513);
  for (IdType i = 13; i < num_unique; ++i) {
    EXPECT_EQ(check_set.count(data_unique[i]), 1);
  }

  for (IdType i = 0; i < num_unique; ++i) {
    std::cout << data_unique[i] << " ";
  }
  std::cout << std::endl;

  CUDA_CALL(cudaFree(data_unique));

  } // check functions:
    //   FillWithDuplicates, NumItems, FillWithDupMutable, CopyUnique

  // TODO: check FillNeighbours, FillUniques
}

TEST(SamgraphTest, TestCudaOrderedHashTable) {
  _TestOrderedHashTable();
}
