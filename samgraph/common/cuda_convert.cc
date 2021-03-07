#include <cusparse.h>

#include "cuda_convert.h"
#include "logging.h"

namespace {

void SortCoo() {

}

} // namespace

namespace samgraph {
namespace common {
namespace cuda {

void ConvertCoo2Csr() {
    cusparseHandle_t handle;
    cusparseMatDescr_t decsr_coo;
    cusparseMatDescr_t decsr_csr;


    CUSPARSE_CALL(cusparseCreate(&handle));

    CUSPARSE_CALL(cusparseCreateMatDescr(&decsr_coo));
    CUSPARSE_CALL(cusparseSetMatType(decsr_coo, CUSPARSE_MATRIX_TYPE_GENERAL));
    CUSPARSE_CALL(cusparseSetMatIndexBase(decsr_coo, CUSPARSE_INDEX_BASE_ZERO));

    // CUSPARSE_CALL(cusparseXcoosortByRow(handle, ));

    // CUSPARSE_CALL(cusparseXcoo2csr(handle, CUSPARSE_INDEX_BASE_ZERO));
}

} // namespace cuda
} // namespace common
} // namespace samgraph
