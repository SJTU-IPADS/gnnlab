#ifndef TESTS_COMMON_H
#define TESTS_COMMON_H

#include <iostream>

#define CUDA_CALL(func)                                            \
  {                                                                \
    cudaError_t e = (func);                                        \
    ASSERT_TRUE(e == cudaSuccess || e == cudaErrorCudartUnloading) \
        << "CUDA: " << cudaGetErrorString(e);                      \
  }

#define GPU_TILE_SIZE 1024
#define GPU_BLOCK_SIZE 256

#define ANSI_TXT_GRN "\033[0;32m"
#define ANSI_TXT_MGT "\033[0;35m"  // Magenta
#define ANSI_TXT_DFT "\033[0;0m"   // Console default
#define GTEST_BOX "[    LOG   ] "
#define COUT_GTEST ANSI_TXT_GRN << GTEST_BOX  // You could add the Default
#define COUT_GTEST_MGT COUT_GTEST << ANSI_TXT_MGT

#define LOG std::cout << COUT_GTEST_MGT

#endif  // TESTS_COMMON_H