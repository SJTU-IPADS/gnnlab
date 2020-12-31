#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <iostream>
#include <limits>
#include <vector>

#include "sampling/cpu/id_hash_map.hpp"
#include "sampling/cpu/graph_storage.hpp"

#define PAGE_SIZE 4096

struct Block {
    uint32_t len;
    uint32_t *row_ptr;
    uint32_t *col_ptr;

    uint32_t seeds_len;
    uint32_t *seeds_ptr;
    COO *coo_ptr;
    CSR *csr_ptr, csc_ptr;
};
int main() {
    std::vector<int> fds;
    std::vector<Block> blocks;
    IdHashMap id_table;
    uint32_t *seeds_ptr = nullptr;

    // Read files
    for (int i = 0; i < 3; i++) {
        std::string row_path = "/graph-learning/dgl/blocks/block-row" + std::to_string(i);
        std::string col_path = "/graph-learning/dgl/blocks/block-col" + std::to_string(i);
        int row_fd = open(row_path.c_str(), O_RDONLY, 0);
        int col_fd = open(col_path.c_str(), O_RDONLY, 0);
        fds.push_back(row_fd);
        fds.push_back(col_fd);

        // Calculate the number of edges
        struct stat st;
        stat(row_path.c_str(), &st);
        uint32_t len = st.st_size / 4;

        // MMIO Mapping
        uint32_t *row_ptr = (uint32_t *) mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE|MAP_FILE, row_fd, 0);
        uint32_t *col_ptr= (uint32_t *) mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE|MAP_FILE, col_fd, 0);

        // Make sure that all data have been loaded into memory.
        // Trigger all the page faults.
        uint32_t magic = 0;
        for (uint32_t i = 0; i < len; i += PAGE_SIZE) {
            magic += row_ptr[i];
            magic += col_ptr[i];
        }

        // Construct adj
        Block block;
        block.len = len;
        block.row_ptr = row_ptr;
        block.col_ptr = col_ptr;

        if (i == 2) {
            std::string seeds_path = "/graph-learning/dgl/blocks/block-seeds2";
            int seeds_fd = open(seeds_path.c_str(), O_RDONLY, 0);
            fds.push_back(seeds_fd);

            struct stat st;
            stat(seeds_path.c_str(), &st);
            uint32_t len = st.st_size / 4;

            seeds_ptr = (uint32_t *) mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE|MAP_FILE, seeds_fd, 0);

            uint32_t magic = 0;
            for (uint32_t i = 0; i < len; i += PAGE_SIZE) {
                magic += seeds_ptr[i];
            }

            block.seeds_len = len;
            block.seeds_ptr = seeds_ptr;

            std::cout << "Loaded seeds with magic number " << magic << std::endl;
        } else {
            block.seeds_len = 0;
            block.seeds_ptr = nullptr;
        }

        blocks.push_back(block);

        std::cout << "Loaded block " << i << " with magic number " << magic
                  << " and " << len << " edges" << std::endl;
    }

    uint32_t default_val = std::numeric_limits<uint32_t>::max();

    for (int i = 2; i >= 0; i--) {
        // Create id table mapping
        auto tic0 = std::chrono::system_clock::now();
        if (i == 2) {
            id_table.Update(blocks[i].seeds_ptr, blocks[i].seeds_len);
        }
        uint32_t num_dst_nodes =id_table.Size();
        id_table.Update(blocks[i].row_ptr, blocks[i].len);
        uint32_t num_src_nodes =id_table.Size();
        auto toc0 = std::chrono::system_clock::now();

        // Construct COO formats
        auto tic1 = std::chrono::system_clock::now();
        COO *coo = new COO();
        coo->num_rows = num_src_nodes;
        coo->num_cols = num_dst_nodes;
        coo->row_ptr = id_table.Map(blocks[i].row_ptr, blocks[i].len, default_val);
        coo->col_ptr = id_table.Map(blocks[i].col_ptr, blocks[i].len, default_val);
        coo->num_cols = blocks[i].len;
        blocks[i].coo_ptr = coo;
        auto toc1 = std::chrono::system_clock::now();

        // Construct CSR formats
        auto tic2 = std::chrono::system_clock::now();
        blocks[i].csr_ptr = COOToCSR(coo);
        auto toc2 = std::chrono::system_clock::now();

        // Construct CSC formats
        auto tic3 = std::chrono::system_clock::now();
        blocks[i].csr_ptr = TransposeCSR(blocks[i].csr_ptr);
        auto toc3 = std::chrono::system_clock::now();

        std::chrono::duration<double> duration_mapping = toc0 - tic0;
        std::chrono::duration<double> duration_coo = toc1 - tic1;
        std::chrono::duration<double> duration_csr = toc2 - tic2;
        std::chrono::duration<double> duration_csc = toc3 - tic3;
        std::chrono::duration<double> duration_total = toc3 - tic0;

        printf("Block %d, mapping: %.4f, coo: %.4f, csr: %.4f, csc: %.4f, total: %4f, src %u, dst %u\n",
               i, duration_mapping.count(), duration_coo.count(), duration_csr.count(), duration_csc.count(),
               duration_total.count(), num_src_nodes, num_dst_nodes);
    }

    // Close files
    for (int fd : fds) {
        close(fd);
    }

    return 0;
}
