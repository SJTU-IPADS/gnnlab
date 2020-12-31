#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <algorithm>

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
    const int64_t N = coo->num_rows;
    const int64_t NNZ = coo->num_edges;
    const uint32_t* row_data = coo->row_ptr;
    const uint32_t* col_data = coo->col_ptr;
    
    uint32_t *indptr = (uint32_t *)malloc((N + 1) * sizeof(uint32_t));
    uint32_t *indices = (uint32_t *)malloc((coo->num_edges * sizeof(uint32_t)));
    
    bool row_sorted = coo->row_sorted;
    bool col_sorted = coo->col_sorted;

    if (row_sorted) {
        // compute indptr
        indptr[0] = 0;
        uint32_t j = 0;
        for (uint32_t i = 0; i < N; ++i) {
            const uint32_t k = j;
            for (; j < NNZ && row_data[j] == i; ++j) {}
            indptr[i + 1] = indptr[i] + j - k;
        }

        memcpy(indices, col_data, coo->num_edges * sizeof(uint32_t));
    } else {
        // compute indptr
        uint32_t *Bp = indptr;
        *(Bp) = 0;
        std::fill(Bp, Bp + N, 0);
        for (int64_t i = 0; i < NNZ; ++i) {
            Bp[row_data[i]]++;
        }

        // cumsum
        for (uint32_t i = 0, cumsum = 0; i < N; ++i) {
            const uint32_t temp = Bp[i];
            Bp[i] = cumsum;
            cumsum += temp;
        }

        uint32_t *Bi = indices;

        // compute indices and data
        for (int64_t i = 0; i < NNZ; ++i) {
            const uint32_t r = row_data[i];
            Bi[Bp[r]] = col_data[i];
            Bp[r]++;
        }
    }

    CSR *csr = new CSR();
    csr->num_rows = coo->num_rows;
    csr->num_cols = coo->num_cols;
    csr->indptr = indptr;
    csr->indices = indices;
    csr->sorted = coo->col_sorted;

    return csr;
}

CSR* TransposeCSR(CSR *csr) {

}
