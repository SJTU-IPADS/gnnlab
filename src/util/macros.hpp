#pragma once

#include <cstdlib>
#include <cstdio>

#include <cuda.h>
#include <cudnn.h>

// Define some error checking macros.
#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(CUresult stat, const char *file, int line) {
   if (stat != CUDA_SUCCESS) {
      const char *error_str;
      cuGetErrorString(stat, &error_str);
      fprintf(stderr, "CUDA Error: %s %s %d\n", error_str, file, line);
      exit(1);
   }
}

#define cudnnErrCheck(stat) { cudnnErrCheck_((stat), __FILE__, __LINE__); }
void cudnnErrCheck_(cudnnStatus_t stat, const char *file, int line) {
   if (stat != CUDNN_STATUS_SUCCESS) {
      fprintf(stderr, "cuDNN Error: %s %s %d\n", cudnnGetErrorString(stat), file, line);
      exit(1);
   }
}
