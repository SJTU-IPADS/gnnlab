#ifndef TESTS_COMMON_H
#define TESTS_COMMON_H

#include <iostream>
#include <cstdlib>

#include <cuda_runtime.h>
#include <cusparse.h>


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

#endif // TESTS_COMMON_H