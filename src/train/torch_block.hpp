#pragma once

#include <vector>
#include <memory>
#include <string>

#include <torch/torch.h>

#include "sampling/cpu/block.hpp"

struct TorchCSR {
    torch::Tensor indptr;
    torch::Tensor indices;

    // CPU meta object list [m, n, nnz]
    torch::Tensor meta;
};

struct TorchBlocks {
    torch::Tensor feature;
    torch::Tensor label;

    std::vector<TorchCSR> csr;
    std::vector<TorchCSR> csc;
    std::vector<torch::Tensor> val;
};

TorchBlocks CPUBlocksToDevice (std::vector<std::shared_ptr<Block>> blocks, std::string device) {
    TorchBlocks torch_blocks;

    torch_blocks.feature = torch::from_blob(
        blocks.front()->block_features,
        {(long long)blocks.front()->num_src_nodes, blocks.front()->feat_dim},
        torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false)
    ).to(device);
    
    torch_blocks.label = torch::from_blob(
        blocks.back()->block_label,
        {(long long)blocks.back()->num_dst_nodes, 1},
        torch::TensorOptions().dtype(torch::kLong).requires_grad(false)
    ).to(device);

    for (int i = 0; i < blocks.size(); i++) {
        std::shared_ptr<CSR> csr = blocks[i]->csr_ptr;
        std::shared_ptr<CSR> csc = blocks[i]->csc_ptr;

        torch_blocks.csr[i].indptr = torch::from_blob(
            csr->indptr,
            {(long long)csr->num_rows + 1},
            torch::TensorOptions().dtype(torch::kInt32).requires_grad(false)
        ).to(device);

        torch_blocks.csr[i].indices = torch::from_blob(
            csr->indices,
            {(long long)csr->num_edges},
            torch::TensorOptions().dtype(torch::kInt32).requires_grad(false)
        ).to(device);

        torch::Tensor val = torch::ones(
            {(long long)csr->num_edges},
            torch::TensorOptions().dtype(torch::kFloat32).device(device).requires_grad(false)
        );

        torch_blocks.csr[i]
    }

    return torch_blocks;
}
