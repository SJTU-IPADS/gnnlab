#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdlib>

#include "test_common/common.h"
#include "test_common/timer.h"

int n_iters = 10;
size_t copy_nbytes = 500ull * 1024 * 1024;

TEST(MemcpyTest, HostMalloc) {
  double pinned_time = 0, time = 0;

  void *host_pinned_data, *host_unppined_data, *dev_data;
  host_unppined_data = malloc(copy_nbytes);
  CUDA_CALL(
      cudaHostAlloc(&host_pinned_data, copy_nbytes, cudaHostAllocDefault));
  CUDA_CALL(cudaMalloc(&dev_data, copy_nbytes));

  for (int i = 0; i < n_iters; i++) {
    Timer t0;
    CUDA_CALL(cudaMemcpy(dev_data, host_pinned_data, copy_nbytes,
                         cudaMemcpyHostToDevice));
    pinned_time += t0.Passed();

    Timer t1;
    CUDA_CALL(cudaMemcpy(dev_data, host_unppined_data, copy_nbytes,
                         cudaMemcpyHostToDevice));
    time += t1.Passed();
  }

  CUDA_CALL(cudaFreeHost(host_pinned_data));
  CUDA_CALL(cudaFree(dev_data));
  free(host_unppined_data);

  LOG << "pinned: " << pinned_time / n_iters
      << " | unpinned: " << time / n_iters << "\n";
}