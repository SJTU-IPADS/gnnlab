#pragma once

#include <cstdint>

struct COO {
    uint32_t num_rows, num_cols;
    uint32_t num_edges;
    uint32_t *row_ptr, *col_ptr;
    bool row_sorted = false, col_sorted = false;
};

struct CSR {
    uint32_t num_rows, num_cols;
    uint32_t num_edges;
    uint32_t *indptr, *indices;
    bool sorted = false;
};

CSR* COOToCSR(COO *coo) {
    return nullptr;
}

CSR* TransposeCSR(CSR *csr) {
    return nullptr;
}
