#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <cusparse.h>

#include "common.h"

constexpr int kEmptyKey = -1;
constexpr size_t kTileSize = 5;
constexpr size_t kBlockSize = 5;

template <typename T>
struct BlockPrefixCallbackOp {
    size_t _running_total;
    
    __device__ BlockPrefixCallbackOp(const T running_total)
        : _running_total(running_total) {}
    
    __device__ T operator()(const T block_aggregate) {
        const T old_prefix = _running_total;
        _running_total += block_aggregate;
        return old_prefix;
      }
};

template<size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void
count_edge(int *edge_src, size_t *item_prefix, const size_t num_input, const size_t fanout) {
    assert(BLOCK_SIZE == blockDim.x);
    
    using BlockReduce = typename cub::BlockReduce<size_t, BLOCK_SIZE>;
    
    const size_t block_start = TILE_SIZE * blockIdx.x;
    const size_t block_end = TILE_SIZE * (blockIdx.x + 1);
    
    size_t count = 0;
    #pragma unroll
    for (size_t index = threadIdx.x + block_start; index < block_end; index += BLOCK_SIZE) {
        if (index < num_input) {
            for (size_t j = 0; j < fanout; j++) {
                if (edge_src[index * fanout + j] != kEmptyKey) {
                    ++count;
                }
            }
        }
    }

    __shared__ typename BlockReduce::TempStorage temp_space;

    count = BlockReduce(temp_space).Sum(count);

    if (threadIdx.x == 0) {
        item_prefix[blockIdx.x] = count;
        if (blockIdx.x == 0) {
            item_prefix[gridDim.x] = 0;
        }
    }
}

template<size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void
compact_edge(const int *tmp_src, const int *tmp_dst, int *out_src, int *out_dst,
             size_t *num_out, const size_t *item_prefix, const size_t num_input, const size_t fanout) {
    assert(BLOCK_SIZE == blockDim.x);

    using BlockScan = typename cub::BlockScan<size_t, BLOCK_SIZE>;
 
    constexpr const size_t VALS_PER_THREAD = TILE_SIZE / BLOCK_SIZE;

    __shared__ typename BlockScan::TempStorage temp_space;

    const size_t offset = item_prefix[blockIdx.x];

    BlockPrefixCallbackOp<size_t> prefix_op(0);

    // count successful placements
    for (size_t i = 0; i < VALS_PER_THREAD; ++i) {
        const size_t index = threadIdx.x + i * BLOCK_SIZE + blockIdx.x * TILE_SIZE;

        size_t item_per_thread = 0;
        if (index < num_input) {
            for (size_t j = 0; j < fanout; j++) {
                if (tmp_src[index * fanout + j] != kEmptyKey) {
                    item_per_thread++;
                }
            }
        }

        size_t item_prefix_per_thread = item_per_thread;
        BlockScan(temp_space).ExclusiveSum(item_prefix_per_thread, item_prefix_per_thread, prefix_op);
        __syncthreads();
  
        
        for (size_t j = 0; j < item_per_thread; j++) {
            out_src[offset + item_prefix_per_thread + j] = tmp_src[index * fanout + j];
            out_dst[offset + item_prefix_per_thread + j] = tmp_dst[index * fanout + j];
        }
    }

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *num_out = item_prefix[gridDim.x];
    }
}

int main() {
    constexpr size_t num_input = 28;
    constexpr size_t fanout = 5;

    int tmp_src[num_input * fanout] = {1, 2, 3, kEmptyKey, kEmptyKey,
                                       1, 2, 3, kEmptyKey, kEmptyKey,
                                       1, 2, 3, kEmptyKey, kEmptyKey,
                                       1, 2, 3, kEmptyKey, kEmptyKey,
                                       1, 2, 3, kEmptyKey, kEmptyKey,
                                       1, 2, 3, kEmptyKey, kEmptyKey,
                                       1, 2, 3, kEmptyKey, kEmptyKey,
                                       1, 2, 3, kEmptyKey, kEmptyKey,
                                       1, 2, 3, kEmptyKey, kEmptyKey,
                                       1, 2, 3, kEmptyKey, kEmptyKey,
                                       1, 2, 3, kEmptyKey, kEmptyKey,
                                       1, 2, 3, kEmptyKey, kEmptyKey,
                                       1, 2, 3, kEmptyKey, kEmptyKey,
                                       1, 2, 3, kEmptyKey, kEmptyKey,
                                       1, 2, 3, kEmptyKey, kEmptyKey,
                                       1, 2, 3, kEmptyKey, kEmptyKey,
                                       1, 2, 3, kEmptyKey, kEmptyKey,
                                       1, 2, 3, kEmptyKey, kEmptyKey,
                                       1, 2, 3, kEmptyKey, kEmptyKey,
                                       1, 2, 3, kEmptyKey, kEmptyKey,
                                       1, 2, 3, kEmptyKey, kEmptyKey,
                                       1, 2, 3, kEmptyKey, kEmptyKey,
                                       1, 2, 3, kEmptyKey, kEmptyKey,
                                       1, 2, 3, kEmptyKey, kEmptyKey,
                                       1, 2, 3, kEmptyKey, kEmptyKey,
                                       1, 2, 3, kEmptyKey, kEmptyKey,
                                       1, 2, 3, kEmptyKey, kEmptyKey,
                                       1, 2, 3, kEmptyKey, kEmptyKey
                                      };
    int tmp_dst[num_input * fanout] = {1, 2, 3, kEmptyKey, kEmptyKey,
                                       1, 2, 3, kEmptyKey, kEmptyKey,
                                       1, 2, 3, kEmptyKey, kEmptyKey,
                                       1, 2, 3, kEmptyKey, kEmptyKey,
                                       1, 2, 3, kEmptyKey, kEmptyKey,
                                       1, 2, 3, kEmptyKey, kEmptyKey,
                                       1, 2, 3, kEmptyKey, kEmptyKey,
                                       1, 2, 3, kEmptyKey, kEmptyKey,
                                       1, 2, 3, kEmptyKey, kEmptyKey,
                                       1, 2, 3, kEmptyKey, kEmptyKey,
                                       1, 2, 3, kEmptyKey, kEmptyKey,
                                       1, 2, 3, kEmptyKey, kEmptyKey,
                                       1, 2, 3, kEmptyKey, kEmptyKey,
                                       1, 2, 3, kEmptyKey, kEmptyKey,
                                       1, 2, 3, kEmptyKey, kEmptyKey,
                                       1, 2, 3, kEmptyKey, kEmptyKey,
                                       1, 2, 3, kEmptyKey, kEmptyKey,
                                       1, 2, 3, kEmptyKey, kEmptyKey,
                                       1, 2, 3, kEmptyKey, kEmptyKey,
                                       1, 2, 3, kEmptyKey, kEmptyKey,
                                       1, 2, 3, kEmptyKey, kEmptyKey,
                                       1, 2, 3, kEmptyKey, kEmptyKey,
                                       1, 2, 3, kEmptyKey, kEmptyKey,
                                       1, 2, 3, kEmptyKey, kEmptyKey,
                                       1, 2, 3, kEmptyKey, kEmptyKey,
                                       1, 2, 3, kEmptyKey, kEmptyKey,
                                       1, 2, 3, kEmptyKey, kEmptyKey,
                                       1, 2, 3, kEmptyKey, kEmptyKey};
    int out_src [num_input * fanout];
    int out_dst [num_input * fanout];
    size_t num_out;
    size_t item_prefix[num_input + 1];

    int *d_tmp_src;
    int *d_tmp_dst;
    int *d_out_src;
    int *d_out_dst;
    size_t *d_num_out;
    size_t *d_item_prefix;

    CUDA_CALL(cudaMalloc(&d_tmp_src, sizeof(int) * num_input * fanout));
    CUDA_CALL(cudaMalloc(&d_tmp_dst, sizeof(int) * num_input * fanout));
    CUDA_CALL(cudaMalloc(&d_out_src, sizeof(int) * num_input * fanout));
    CUDA_CALL(cudaMalloc(&d_out_dst, sizeof(int) * num_input * fanout));
    CUDA_CALL(cudaMalloc(&d_num_out, sizeof(size_t)));
    CUDA_CALL(cudaMalloc(&d_item_prefix, sizeof(size_t) * (num_input + 1)));

    CUDA_CALL(cudaMemcpy(d_tmp_src, tmp_src, sizeof(int) * num_input * fanout, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_tmp_dst, tmp_dst, sizeof(int) * num_input * fanout, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_item_prefix, item_prefix, sizeof(size_t) * (num_input + 1), cudaMemcpyHostToDevice));

    const size_t num_tiles = (num_input + kTileSize - 1) / kTileSize;
    const dim3 grid(num_tiles);
    const dim3 block(kBlockSize);

    count_edge<kBlockSize, kTileSize>
        <<<grid, block>>>(d_tmp_src, d_item_prefix, num_input, fanout);
    CUDA_CALL(cudaGetLastError());

    size_t workspace_bytes;
    CUDA_CALL(cub::DeviceScan::ExclusiveSum(
        nullptr, workspace_bytes, static_cast<size_t *>(nullptr),
        static_cast<size_t *>(nullptr), grid.x + 1));
        
    void *d_workspace;
    CUDA_CALL(cudaMalloc(&d_workspace, workspace_bytes));
    CUDA_CALL(cub::DeviceScan::ExclusiveSum(
        d_workspace, workspace_bytes, d_item_prefix, d_item_prefix, grid.x + 1));

    compact_edge<kBlockSize, kTileSize>
        <<<grid, block>>>(d_tmp_src, d_tmp_src, d_out_src, d_out_dst, d_num_out, d_item_prefix, num_input, fanout);
    CUDA_CALL(cudaGetLastError());
    
    CUDA_CALL(cudaMemcpy(out_src, d_out_src, sizeof(int) * num_input * fanout, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(out_dst, d_out_dst, sizeof(int) * num_input * fanout, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(&num_out, d_num_out, sizeof(size_t), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(item_prefix, d_item_prefix, sizeof(size_t) * (num_input), cudaMemcpyDeviceToHost));

    std::cout << num_tiles << '\t' << kBlockSize <<std::endl;

    for (int i = 0; i < (grid.x + 1); i++) {
        std::cout << item_prefix[i] << ' ';
    }
    std::cout << std::endl;

    for (int i = 0; i < (num_input * fanout); i++) {
        std::cout << out_src[i] << ' ';
    }
    std::cout << std::endl;
    for (int i = 0; i < (num_input * fanout); i++) {
        std::cout << out_dst[i] << ' ';
    }
    std::cout << std::endl;

    std::cout << num_out << std::endl;

    CUDA_CALL(cudaFree(d_workspace));
    CUDA_CALL(cudaFree(d_tmp_src));
    CUDA_CALL(cudaFree(d_tmp_dst));
    CUDA_CALL(cudaFree(d_out_src));
    CUDA_CALL(cudaFree(d_out_dst));
    CUDA_CALL(cudaFree(d_num_out));
    CUDA_CALL(cudaFree(d_item_prefix));

    CUDA_CALL(cudaDeviceReset());

    return 0;
}