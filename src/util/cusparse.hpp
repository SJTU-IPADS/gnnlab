#pragma once

#include <cstdlib>
#include <cusparse.h>

#define cusparseErrCheck(stat) { cusparseErrCheck_((stat), __FILE__, __LINE__); }
void cusparseErrCheck_(cusparseStatus_t stat, const char *file, int line) {
   if (stat != CUSPARSE_STATUS_SUCCESS) {
      fprintf(stderr, "cuSparse Error: %s %s %d\n", cusparseGetErrorString(stat), file, line);
      exit(1);
   }
}

cusparseHandle_t *myCusparseHandle() {
    static thread_local cusparseHandle_t *handle = nullptr;
    if (!handle) {
        cusparseErrCheck(cusparseCreate(handle));
    }
    return handle;
}
