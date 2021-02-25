#ifndef TEST
#define TEST

#include <cstdlib>
#include <cstdio>

#include <cuda_runtime.h>
#include <cusparse.h>

// Define some error checking macros.
#define CUDA_CALL(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
   if (stat != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
      exit(1);
   }
}

#define CUSPARSE_CALL(stat) { cusparseErrCheck_((stat), __FILE__, __LINE__); }
void cusparseErrCheck_(cusparseStatus_t stat, const char *file, int line) {
   if (stat != CUSPARSE_STATUS_SUCCESS) {
      fprintf(stderr, "cuSparse Error: %s %s %d\n", cusparseGetErrorString(stat), file, line);
      exit(1);
   }
}

cusparseHandle_t *cusparse_handle() {
   static thread_local bool initialized = false;
   static thread_local cusparseHandle_t handle;
   if (!initialized) {
      initialized = true;
      CUSPARSE_CALL(cusparseCreate(&handle));
   }
   return &handle;
}

#endif
