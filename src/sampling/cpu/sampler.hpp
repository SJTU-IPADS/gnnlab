#pragma once

#include <algorithm>
#include <vector>

#include "sampling/cpu/graph_storage.hpp"
#include "sampling/cpu/random.hpp"

COO *SampleBlock(CSR *csr, const std::vector<uint32_t> &rows, int num_picks){
    uint32_t *indptr = csr->indptr;
    uint32_t *indices = csr->indices;

    uint32_t num_rows = rows.size();
    uint32_t num_edges = num_rows * num_picks;

    uint32_t *picked_row = (uint32_t *) malloc(num_rows * num_picks * sizeof(uint32_t));
    uint32_t *picked_col = (uint32_t *) malloc(num_picks * num_rows * sizeof(uint32_t));

    bool all_has_fanout = true;
#pragma omp parallel for reduction(&&:all_has_fanout)
    for (uint32_t i = 0; i < num_rows; ++i) {
        uint32_t rid = rows[i];
        uint32_t len = indptr[rid + 1] - indptr[rid];

        all_has_fanout = all_has_fanout && (len >= num_picks);
    }

#pragma omp parallel for
    for (uint32_t i = 0; i < num_rows; ++i) {
        const uint32_t rid = rows[i];
        const uint32_t off = indptr[rid];
        const uint32_t len = indptr[rid + 1] - off;

        if (len <= num_picks) {
            uint32_t j = 0;
            for (; j < len; ++j){
                picked_row[i * num_picks + j] = rid;
                picked_col[i * num_picks + j] = indices[off + j];
            }

            for (; j < num_picks; ++j) {
                picked_row[i * num_picks + j] = -1;
                picked_col[i * num_picks + j] = -1;
            }
        } else {
            // reservoir algorithm
            // time: O(population), space: O(num)
            for (uint32_t j = 0; j < num_picks; ++j) {
                picked_row[i * num_picks + j] = rid;
            }

            for (uint32_t j = 0; j < num_picks; ++j) {
                picked_col[i * num_picks + j] = indices[off + j];
            }

            for (uint32_t j = num_picks; j < len; ++j) {
                const uint32_t k = RandInt(0, j + 1);
                if (k < num_picks) {
                    picked_col[i * num_picks + k] = indices[off + j];
                }
            }
        }
    }

    if (!all_has_fanout) {
        uint32_t *row_end = std::remove_if(picked_row, picked_row + num_rows * num_picks,
                                           [](uint32_t i) { return i == -1; });
        uint32_t *col_end = std::remove_if(picked_col, picked_col + num_rows * num_picks,
                                           [](uint32_t i) { return i == -1; });

        num_edges = row_end - picked_row;
    }

    COO *coo = new COO();
    coo->num_rows = csr->num_rows;
    coo->num_cols = csr->num_cols;
    coo->num_edges = num_edges;
    coo->row_ptr = picked_row;
    coo->col_ptr = picked_col;
    return coo;
}


COO *SampleNeighbors(CSR *csr, const std::vector<uint32_t> &seeds, const std::vector<int> &fanout) {

    return nullptr;
}
