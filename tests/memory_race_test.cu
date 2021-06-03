#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "test_common/common.h"
#include "test_common/timer.h"

constexpr size_t memory_range = 300;
constexpr int num_iter = 10;

namespace {

template <size_t TILE_SIZE, size_t BLOCK_SIZE>
__global__ void memory_access(float *A, float *B, size_t num_items) {

  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_items) {
      for (size_t i = 0; i < memory_range && (index + i) < num_items; i++) {
        B[index] *= A[index + i];
      }
    }
  }
}

} // namespace

TEST(MemoryRaceTest, SequantialTest) {
  float *cpu_A, *cpu_B;
  size_t num_items = 1 << 24;

  cpu_A = (float *)malloc(sizeof(float) * num_items);
  cpu_B = (float *)malloc(sizeof(float) * num_items);

  std::vector<float *> gpu_As(num_iter);
  std::vector<float *> gpu_Bs(num_iter);

  for (int i = 0; i < num_iter; i++) {
    CUDA_CALL(cudaMalloc(&gpu_As[i], sizeof(float) * num_items));
    CUDA_CALL(cudaMalloc(&gpu_Bs[i], sizeof(float) * num_items));
  }

  double copy_time = 0;
  double cal_time = 0;
  double total_time = 0;

  cudaStream_t stream;
  CUDA_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  const size_t num_tiles = (num_items + GPU_TILE_SIZE - 1) / GPU_TILE_SIZE;
  dim3 grid(num_tiles);
  dim3 block(GPU_BLOCK_SIZE);

  for (int i = 0; i < num_iter; i++) {
    Timer t0;

    float *gpu_A = gpu_As[i];
    float *gpu_B = gpu_Bs[i];

    CUDA_CALL(cudaMemcpyAsync(gpu_A, cpu_A, sizeof(float) * num_items,
                              cudaMemcpyHostToDevice, stream));
    CUDA_CALL(cudaMemcpyAsync(gpu_B, cpu_B, sizeof(float) * num_items,
                              cudaMemcpyHostToDevice, stream));
    CUDA_CALL(cudaStreamSynchronize(stream));
    copy_time += t0.Passed();

    Timer t1;
    memory_access<GPU_TILE_SIZE, GPU_BLOCK_SIZE>
        <<<grid, block, 0, stream>>>(gpu_A, gpu_B, num_items);

    CUDA_CALL(cudaStreamSynchronize(stream));
    cal_time += t1.Passed();
    total_time += t0.Passed();
  }

  for (int i = 0; i < num_iter; i++) {
    CUDA_CALL(cudaFree(gpu_As[i]));
    CUDA_CALL(cudaFree(gpu_Bs[i]));
  }

  CUDA_CALL(cudaStreamDestroy(stream));

  free(cpu_A);
  free(cpu_B);

  LOG << "copy: " << copy_time / num_iter << " | cal: " << cal_time / num_iter
      << " | total: " << total_time / num_iter << ANSI_TXT_DFT << "\n";
}

TEST(MemoryRaceTest, PipelineEventTest) {
  float *cpu_A, *cpu_B;
  size_t num_items = 1 << 24;

  cpu_A = (float *)malloc(sizeof(float) * num_items);
  cpu_B = (float *)malloc(sizeof(float) * num_items);

  std::vector<float *> gpu_As(num_iter);
  std::vector<float *> gpu_Bs(num_iter);

  double total_time = 0;

  cudaStream_t copy_stream, comp_stream;
  CUDA_CALL(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking));
  CUDA_CALL(cudaStreamCreateWithFlags(&comp_stream, cudaStreamNonBlocking));

  CUDA_CALL(cudaStreamSynchronize(copy_stream));
  CUDA_CALL(cudaStreamSynchronize(comp_stream));

  std::vector<cudaEvent_t> events(num_iter);

  for (int i = 0; i < num_iter; i++) {
    CUDA_CALL(cudaMalloc(&gpu_As[i], sizeof(float) * num_items));
    CUDA_CALL(cudaMalloc(&gpu_Bs[i], sizeof(float) * num_items));
    CUDA_CALL(cudaEventCreate(&events[i]));
  }

  const size_t num_tiles = (num_items + GPU_TILE_SIZE - 1) / GPU_TILE_SIZE;
  dim3 grid(num_tiles);
  dim3 block(GPU_BLOCK_SIZE);

  Timer t;
  for (int i = 0; i < num_iter; i++) {
    float *gpu_A = gpu_As[i];
    float *gpu_B = gpu_Bs[i];

    CUDA_CALL(cudaMemcpyAsync(gpu_A, cpu_A, sizeof(float) * num_items,
                              cudaMemcpyHostToDevice, copy_stream));
    CUDA_CALL(cudaMemcpyAsync(gpu_B, cpu_B, sizeof(float) * num_items,
                              cudaMemcpyHostToDevice, copy_stream));
    CUDA_CALL(cudaEventRecord(events[i], copy_stream));

    CUDA_CALL(cudaStreamWaitEvent(comp_stream, events[i], 0));
    memory_access<GPU_TILE_SIZE, GPU_BLOCK_SIZE>
        <<<grid, block, 0, comp_stream>>>(gpu_A, gpu_B, num_items);
  }

  CUDA_CALL(cudaStreamSynchronize(copy_stream));
  CUDA_CALL(cudaStreamSynchronize(comp_stream));
  total_time += t.Passed();

  for (int i = 0; i < num_iter; i++) {
    CUDA_CALL(cudaFree(gpu_As[i]));
    CUDA_CALL(cudaFree(gpu_Bs[i]));
    CUDA_CALL(cudaEventDestroy(events[i]));
  }

  free(cpu_A);
  free(cpu_B);

  CUDA_CALL(cudaStreamDestroy(copy_stream));
  CUDA_CALL(cudaStreamDestroy(comp_stream));

  LOG << "total: " << total_time / num_iter << ANSI_TXT_DFT << "\n";
}

TEST(MemoryRaceTest, PipelineThreadTest) {
  float *cpu_A, *cpu_B;
  size_t num_items = 1 << 24;

  cpu_A = (float *)malloc(sizeof(float) * num_items);
  cpu_B = (float *)malloc(sizeof(float) * num_items);

  std::vector<float *> gpu_As(num_iter);
  std::vector<float *> gpu_Bs(num_iter);

  double copy_time = 0;
  double cal_time = 0;
  double total_time = 0;

  cudaStream_t copy_stream, comp_stream;
  CUDA_CALL(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking));
  CUDA_CALL(cudaStreamCreateWithFlags(&comp_stream, cudaStreamNonBlocking));

  CUDA_CALL(cudaStreamSynchronize(copy_stream));
  CUDA_CALL(cudaStreamSynchronize(comp_stream));

  for (int i = 0; i < num_iter; i++) {
    CUDA_CALL(cudaMalloc(&gpu_As[i], sizeof(float) * num_items));
    CUDA_CALL(cudaMalloc(&gpu_Bs[i], sizeof(float) * num_items));
  }

  const size_t num_tiles = (num_items + GPU_TILE_SIZE - 1) / GPU_TILE_SIZE;
  dim3 grid(num_tiles);
  dim3 block(GPU_BLOCK_SIZE);

  Timer t;
  std::vector<std::thread> threads;
  int task_count = 0;
  std::mutex mu;
  std::condition_variable condition;

  threads.push_back(std::thread([&]() {
    Timer t;
    for (int i = 0; i < num_iter; i++) {
      float *gpu_A = gpu_As[i];
      float *gpu_B = gpu_Bs[i];

      CUDA_CALL(cudaMemcpyAsync(gpu_A, cpu_A, sizeof(float) * num_items,
                                cudaMemcpyHostToDevice, copy_stream));
      CUDA_CALL(cudaMemcpyAsync(gpu_B, cpu_B, sizeof(float) * num_items,
                                cudaMemcpyHostToDevice, copy_stream));
      CUDA_CALL(cudaStreamSynchronize(copy_stream));

      {
        std::unique_lock<std::mutex> lock(mu);
        task_count++;
      }
      condition.notify_one();
    }
    copy_time += t.Passed();
  }));

  threads.push_back(std::thread([&]() {
    Timer t;
    for (int i = 0; i < num_iter; i++) {
      {
        std::unique_lock<std::mutex> lock(mu);
        condition.wait(lock, [&] { return task_count; });
      }

      float *gpu_A = gpu_As[i];
      float *gpu_B = gpu_Bs[i];
      memory_access<GPU_TILE_SIZE, GPU_BLOCK_SIZE>
          <<<grid, block, 0, comp_stream>>>(gpu_A, gpu_B, num_items);
    }
    CUDA_CALL(cudaStreamSynchronize(comp_stream));
    cal_time += t.Passed();
  }));

  for (auto &t : threads) {
    t.join();
  }

  total_time += t.Passed();

  for (int i = 0; i < num_iter; i++) {
    CUDA_CALL(cudaFree(gpu_As[i]));
    CUDA_CALL(cudaFree(gpu_Bs[i]));
  }

  free(cpu_A);
  free(cpu_B);

  CUDA_CALL(cudaStreamDestroy(copy_stream));
  CUDA_CALL(cudaStreamDestroy(comp_stream));

  LOG << "copy: " << copy_time / num_iter << " | cal: " << cal_time / num_iter
      << " | total: " << total_time / num_iter << ANSI_TXT_DFT << "\n";
}
