#pragma once

#include <cstdlib>
#include <cstdint>
#include <vector>

#include "sampling/cpu/graph_storage.hpp"

struct Block {
    uint32_t num_src_nodes = 0;
    uint32_t num_dst_nodes = 0;
    int num_picks = 0;
    uint32_t num_nodes = 0;
    int feat_dim = 0;

    std::vector<uint32_t> seeds;
    std::vector<uint32_t> block_nodes;

    uint32_t *block_features = nullptr;
    uint32_t *block_label = nullptr;

    std::shared_ptr<COO> coo_ptr;
    std::shared_ptr<CSR> csr_ptr, csc_ptr;

    ~Block() {
        if (block_features) {
            free(block_features);
        }

        if (block_label) {
            free(block_label);
        }
    }
};
