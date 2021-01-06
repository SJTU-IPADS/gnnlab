#pragma once

#include <memory>
#include <vector>
#include <limits>

#include "data/dataset.hpp"
#include "sampling/cpu/id_hash_map.hpp"
#include "sampling/cpu/shuffler.hpp"
#include "sampling/cpu/graph_storage.hpp"
#include "sampling/cpu/block.hpp"
#include "sampling/cpu/shuffler.hpp"
#include "sampling/cpu/sampling.hpp"

struct SamplingTask {
    int num_blocks = 0;
    std::vector<int> fanout;
};

std::vector<std::shared_ptr<Block>> SampleMultiHops(std::shared_ptr<Dataset> dataset, NodesBatch batch, SamplingTask task) {
    std::vector<std::shared_ptr<Block>> blocks(task.num_blocks);
    std::shared_ptr<CSR> input_graph = dataset->GetCSR();

    uint32_t default_val = std::numeric_limits<uint32_t>::max();

    std::vector<uint32_t> seeds(batch.ids, batch.ids + batch.num_samples);    
    IdHashMap idmap(seeds);
    for (int bid = task.num_blocks - 1; bid >= 0; bid++) {
        // 1. Sampling
        blocks[bid] = SampleBlock(input_graph, seeds, task.fanout[task.num_blocks - bid - 1]);
        std::shared_ptr<Block> block = blocks[bid];

        // 2. Id Remapping
        block->num_dst_nodes = idmap.Size();
        idmap.Update(block->raw_block->col_ptr, block->raw_block->num_edges);
        block->num_src_nodes = idmap.Size();
        idmap.Values(block->nodes2oid);

        // 4. Create Graph format
        block->coo_ptr = std::make_shared<COO>();
        block->coo_ptr->num_rows = block->num_src_nodes;
        block->coo_ptr->num_cols = block->num_dst_nodes;
        block->coo_ptr->row_ptr = idmap.Map(block->raw_block->row_ptr, block->raw_block->num_edges, default_val);
        block->coo_ptr->col_ptr = idmap.Map(block->raw_block->col_ptr, block->raw_block->num_edges, default_val);
        block->coo_ptr->num_cols = blocks.len;

        block->csr_ptr = COOToCSR(block->coo_ptr);

        block->csc_ptr = TransposeCSR(block->csr_ptr);
        
        block->raw_block.reset();
        block->coo_ptr.reset();

        // 5. Select feature and label data

    }

    return blocks;
}
