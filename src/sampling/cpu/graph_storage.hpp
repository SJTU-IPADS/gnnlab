#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <memory>

struct COO {
    size_t num_rows = 0, num_cols = 0;
    size_t num_edges = 0;
    uint32_t *row_ptr = nullptr, *col_ptr = nullptr;
    bool row_sorted = false, col_sorted = false;
    bool need_free = true;

    ~COO() {
        if (need_free && row_ptr) {
            free(row_ptr);
        }

        if (need_free && col_ptr) {
            free(col_ptr);
        }
    }
};

struct CSR {
    size_t num_rows = 0, num_cols = 0;
    size_t num_edges = 0;
    uint32_t *indptr = nullptr, *indices = nullptr;
    bool sorted = false;
    bool need_free = true;

    ~CSR() {
        if (need_free && indptr) {
            free(indptr);
        }

        if (need_free && indices) {
            free(indices);
        }
    }
};

std::shared_ptr<CSR> COOToCSR(std::shared_ptr<COO> coo) {
    const size_t N = coo->num_rows;
    const size_t NNZ = coo->num_edges;
    const uint32_t* row_data = coo->row_ptr;
    const uint32_t* col_data = coo->col_ptr;
    
    uint32_t *indptr = (uint32_t *)malloc((N + 1) * sizeof(uint32_t));
    uint32_t *indices = (uint32_t *)malloc((NNZ * sizeof(uint32_t)));
    
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
        Bp++;
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

    std::shared_ptr<CSR> csr = std::make_shared<CSR>();
    csr->num_rows = coo->num_rows;
    csr->num_cols = coo->num_cols;
    csr->num_edges = coo->num_edges;
    csr->indptr = indptr;
    csr->indices = indices;
    csr->sorted = coo->col_sorted;

    return csr;
}

std::shared_ptr<CSR> TransposeCSR(std::shared_ptr<CSR> csr) {
    const size_t N = csr->num_rows;
    const size_t M = csr->num_cols;
    const size_t nnz = csr->num_edges;
    const uint32_t* Ap = csr->indptr;
    const uint32_t* Aj = csr->indices;
    uint32_t* ret_indptr = (uint32_t *)malloc((M + 1) * sizeof(uint32_t));
    uint32_t* ret_indices =  (uint32_t *)malloc((nnz * sizeof(uint32_t)));
    uint32_t* Bp = ret_indptr;
    uint32_t* Bi = ret_indices;

    std::fill(Bp, Bp + M, 0);

    for (int64_t j = 0; j < nnz; ++j) {
        Bp[Aj[j]]++;
    }

    // cumsum
    for (int64_t i = 0, cumsum = 0; i < M; ++i) {
        const uint32_t temp = Bp[i];
        Bp[i] = cumsum;
        cumsum += temp;
    }
    Bp[M] = nnz;

    for (int64_t i = 0; i < N; ++i) {
        for (uint32_t j = Ap[i]; j < Ap[i+1]; ++j) {
            const uint32_t dst = Aj[j];
            Bi[Bp[dst]] = i;
            Bp[dst]++;
        }
    }

    // correct the indptr
    for (int64_t i = 0, last = 0; i <= M; ++i) {
        uint32_t temp = Bp[i];
        Bp[i] = last;
        last = temp;
    }

    std::shared_ptr<CSR> csc = std::make_shared<CSR>();
    csc->num_rows = csr->num_rows;
    csc->num_cols = csr->num_cols;
    csc->num_edges = csr->num_edges;
    csc->indptr = ret_indptr;
    csc->indices = ret_indices;
    csc->sorted = true;

    return csc;
}
