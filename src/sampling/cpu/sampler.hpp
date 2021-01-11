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
#include "sampling/cpu/index_select.hpp"
#include "util/performance.hpp"
#include "util/tictoc.hpp"

struct SamplingTask {
    int num_blocks = 0;
    std::vector<int> fanout;
};

std::vector<std::shared_ptr<Block>> SampleMultiHops(std::shared_ptr<Dataset> dataset, const NodesBatch batch, const SamplingTask task) {
    Performance &p = Performance::Instance();
    TicToc t;
    t.Tic(0);
    std::vector<std::shared_ptr<Block>> blocks(task.num_blocks);
    std::shared_ptr<CSR> input_graph = dataset->GetCSR();

    uint32_t default_val = std::numeric_limits<uint32_t>::max();

    std::vector<uint32_t> seeds(batch.ids, batch.ids + batch.num_samples);    
    IdHashMap idmap(seeds);
    for (int bid = task.num_blocks - 1; bid >= 0; bid--) {
        t.Tic(1);
        // 1. Sampling
        std::shared_ptr<Block> block = SampleBlock(input_graph, seeds, task.fanout[bid]);
        blocks[bid] = block;
        p.blocks[bid].sample.Log(t.Toc(1));

        // 2. Id Remapping
        t.Tic(2);
        block->num_dst_nodes = idmap.Size();
        idmap.Update(block->raw_block->row_ptr, block->raw_block->num_edges);
        block->num_src_nodes = idmap.Size();
        p.blocks[bid].remap.Log(t.Toc(2));

        // 3. Create Graph format
        t.Tic(3);
        block->coo_ptr = std::make_shared<COO>();
        block->coo_ptr->num_rows = block->num_src_nodes;
        block->coo_ptr->num_cols = block->num_dst_nodes;
        block->coo_ptr->num_edges = block->raw_block->num_edges;
        block->coo_ptr->row_ptr = idmap.Map(block->raw_block->row_ptr, block->raw_block->num_edges, default_val);
        block->coo_ptr->col_ptr = idmap.Map(block->raw_block->col_ptr, block->raw_block->num_edges, default_val);
        p.blocks[bid].create_coo.Log(t.Toc(3));

        t.Tic(4);
        block->csr_ptr = COOToCSR(block->coo_ptr);
        p.blocks[bid].coo2csr.Log(t.Toc(4));

        t.Tic(5);
        block->csc_ptr = TransposeCSR(block->csr_ptr);
        p.blocks[bid].csr2csc.Log(t.Toc(5));
        
        block->raw_block.reset();
        block->coo_ptr.reset();

        if (bid > 0) {
            idmap.Values(seeds);
        } else {
            idmap.Values(blocks[0]->node_index);
        }
        p.blocks[bid].block.Log(t.Toc(1));
    }
    p.sampling.Log(t.Toc(0));

    t.Tic(6);
    blocks[0]->block_features = IndexSelect<float>(dataset->GetFeature().data, dataset->GetFeature().dim, blocks[0]->node_index);
    blocks[task.num_blocks - 1]->block_label = IndexSelect<uint32_t>(dataset->GetLabel().data, 1, blocks[task.num_blocks - 1]->seed_index);
    p.index_select.Log(t.Toc(6));

    return blocks;
}
