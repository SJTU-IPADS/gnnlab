#pragma once

#include <vector>
#include <memory>
#include <string>

#include <torch/torch.h>

#include "sampling/cpu/block.hpp"

struct TorchCSR {
    torch::Tensor indptr;
    torch::Tensor indices;
    torch::Tensor val;

    // CPU meta object list [m, k, nnz]
    int m, k, nnz;
};

struct TorchBlocks {
    torch::Tensor feature;
    torch::Tensor label;

    std::vector<TorchCSR> csr;
    std::vector<TorchCSR> csc;
};

TorchBlocks CPUBlocksToDevice (std::vector<std::shared_ptr<Block>> blocks, std::string device) {
    TorchBlocks torch_blocks;
    torch_blocks.csr.resize(blocks.size());
    torch_blocks.csc.resize(blocks.size());

    torch_blocks.feature = torch::from_blob(
        blocks.front()->block_features,
        {(long long)blocks.front()->num_src_nodes, (long long)blocks.front()->feat_dim},
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

        // 1. Convert CSR
        torch_blocks.csr[i].indptr = torch::from_blob(
            csr->indptr,
            {(long long)csr->num_rows + 1},
            torch::TensorOptions().dtype(torch::kInt32)
        ).to(device);

        torch_blocks.csr[i].indices = torch::from_blob(
            csr->indices,
            {(long long)csr->num_edges},
            torch::TensorOptions().dtype(torch::kInt32)
        ).to(device);

        torch_blocks.csr[i].val = torch::ones(
            {(long long)csr->num_edges},
            torch::TensorOptions().dtype(torch::kFloat32)
        ).to(device);

        torch_blocks.csr[i].m = (int)csr->num_rows;
        torch_blocks.csr[i].k = (int)csr->num_cols;
        torch_blocks.csr[i].nnz = (int)csr->num_edges;

        // 2. Convert CSC
        torch_blocks.csc[i].indptr = torch::from_blob(
            csc->indptr,
            {(long long)csc->num_rows + 1},
            torch::TensorOptions().dtype(torch::kInt32)
        ).to(device);

        torch_blocks.csc[i].indices = torch::from_blob(
            csc->indices,
            {(long long)csc->num_edges},
            torch::TensorOptions().dtype(torch::kInt32)
        ).to(device);

        torch_blocks.csc[i].val = torch::ones(
            {(long long)csc->num_edges},
            torch::TensorOptions().dtype(torch::kFloat32)
        ).to(device);

        torch_blocks.csc[i].m = (int)csc->num_rows;
        torch_blocks.csc[i].k = (int)csc->num_cols;
        torch_blocks.csc[i].nnz = (int)csc->num_edges;
    }

    return torch_blocks;
}
