#pragma once

#include <vector>
#include <memory>

#include "sampling/cpu/random.hpp"
#include "sampling/cpu/block.hpp"
#include "sampling/cpu/graph_storage.hpp"
#include "sampling/cpu/shuffler.hpp"

std::shared_ptr<Block> SampleBlock(std::shared_ptr<CSR> csr, std::vector<uint32_t> &rows, int num_picks) {
    uint32_t *indptr = csr->indptr;
    uint32_t *indices = csr->indices;

    size_t num_rows = rows.size();
    size_t num_edges = num_rows * num_picks;

    uint32_t *picked_row = (uint32_t *) malloc(num_rows * num_picks * sizeof(uint32_t));
    uint32_t *picked_col = (uint32_t *) malloc(num_rows * num_picks * sizeof(uint32_t));

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

    std::shared_ptr<Block> block = std::make_shared<Block>();
    block->seeds = std::move(rows);

    // Exchange row and col
    block->raw_block = std::make_shared<COO>();
    block->raw_block->num_rows = csr->num_cols;
    block->raw_block->num_cols = csr->num_rows;
    block->raw_block->num_edges = num_edges;
    block->raw_block->row_ptr = picked_col;
    block->raw_block->col_ptr = picked_row;

    return block;
}
