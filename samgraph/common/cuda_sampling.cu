#include <chrono>
#include <cassert>

#include <curand.h>
#include <curand_kernel.h>
#include <cub/cub.cuh>

#include "cuda_sampling.h"
#include "config.h"
#include "logging.h"

namespace {

template <typename T>
struct BlockPrefixCallbackOp {
    uint32_t _running_total;
    
    __device__ BlockPrefixCallbackOp(const T running_total)
        : _running_total(running_total) {}
    
    __device__ T operator()(const T block_aggregate) {
        const T old_prefix = _running_total;
        _running_total += block_aggregate;
        return old_prefix;
      }
};

} // namespace 

namespace samgraph {
namespace common {
namespace cuda {

template<int BLOCK_SIZE, size_t TILE_SIZE>
__global__ void
sample(const nodeid_t *indptr, const nodeid_t *indices, const size_t num_node, const size_t num_edge, const nodeid_t *input,
         const size_t num_input, const int fanout, nodeid_t *tmp_src, nodeid_t *tmp_dst, size_t *num_out, unsigned long seed) {
    assert(BLOCK_SIZE == blockDim.x);
    const size_t block_start = TILE_SIZE * blockIdx.x;
    const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    curandState state;
    curand_init(seed, i, 0, &state);

    #pragma unroll
    for (size_t index = threadIdx.x + block_start; index < block_end; index += BLOCK_SIZE) {
        if (index < num_input) {
            const nodeid_t rid = input[index];
            const nodeid_t off = indptr[rid];
            const nodeid_t len = indptr[rid + 1] - indptr[rid];

            if (len <= fanout) {
                size_t j = 0;
                for (; j < len; ++j) {
                    tmp_src[index * fanout + j] = rid;
                    tmp_dst[index * fanout + j] = indices[off + j];
                }

                for (; j < fanout; ++j) {
                    tmp_src[index * fanout + j] = Config::kEmptyKey;
                    tmp_dst[index * fanout + j] = Config::kEmptyKey;
                }
            } else {
                for (size_t j = 0; j < fanout; ++j) {
                    tmp_src[index * fanout + j] = rid;
                    tmp_dst[index * fanout + j] = indices[off + j];
                }

                for (size_t j = fanout; j < len; ++j) {
                    size_t k = curand(&state) % (j + 1);
                    if (k < fanout) {
                        tmp_dst[index * fanout + k] = indices[off + j];
                    }
                } 
            }
        }
    }
}

template<int BLOCK_SIZE, size_t TILE_SIZE>
__global__ void
count_edge(nodeid_t *edge_src, nodeid_t *item_prefix, const size_t num_input, const int fanout) {
    assert(BLOCK_SIZE == blockDim.x);
    
    using BlockReduce = typename cub::BlockReduce<nodeid_t, BLOCK_SIZE>;
    
    const size_t block_start = TILE_SIZE * blockIdx.x;
    const size_t block_end = TILE_SIZE * (blockIdx.x + 1);
    
    uint32_t count = 0;
    #pragma unroll
    for (size_t index = threadIdx.x + block_start; index < block_end; index += BLOCK_SIZE) {
        if (index < num_input) {
            for (int j = 0; j < fanout; j++) {
                if (edge_src[index * fanout + j] != Config::kEmptyKey) {
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

template <int BLOCK_SIZE, size_t TILE_SIZE>
__global__ void 
compact_edge(const nodeid_t *tmp_src, nodeid_t *tmp_dst, nodeid_t *out_src, nodeid_t *out_dst,
             size_t *num_out, const nodeid_t *item_prefix, const size_t num_input, const int fanout) {
    assert(BLOCK_SIZE == blockDim.x);

    using BlockScan = typename cub::BlockScan<uint32_t, BLOCK_SIZE>;
 
    constexpr const int VALS_PER_THREAD = TILE_SIZE / BLOCK_SIZE;

    __shared__ typename BlockScan::TempStorage temp_space;

    const uint32_t offset = item_prefix[blockIdx.x];

    BlockPrefixCallbackOp<uint32_t> prefix_op(0);

    // count successful placements
    for (int i = 0; i < VALS_PER_THREAD; ++i) {
        const uint32_t index = threadIdx.x + i * BLOCK_SIZE + blockIdx.x * TILE_SIZE;

        uint32_t item_per_thread = 0;
        if (index < num_input) {
            for (int j = 0; j < fanout; j++) {
                if (tmp_src[index * fanout + j] != Config::kEmptyKey) {
                    item_per_thread++;
                }
            }
        } else {
            item_per_thread = 0;
        }

        uint32_t item_prefix_per_thread = item_per_thread;
        BlockScan(temp_space).ExclusiveSum(item_prefix_per_thread, item_prefix_per_thread, prefix_op);
        __syncthreads();
  
        if (item_per_thread > 0) {
            for (int j = 0; j < item_per_thread; j++) {
                out_src[offset + item_prefix_per_thread + j] = tmp_src[index * fanout + j];
                out_dst[offset + item_prefix_per_thread + j] = tmp_dst[index * fanout + j];
            }
        }
    }

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *num_out = item_prefix[gridDim.x];
    }
}

void DeviceSample(const nodeid_t *indptr, const nodeid_t *indices, const size_t num_node, const size_t num_edge, const nodeid_t *input,
                  const size_t num_input, const int fanout, nodeid_t *out_src, nodeid_t *out_dst, size_t *num_out, cudaStream_t stream) {
    const uint32_t num_tiles = (num_node + Config::kCudaTileSize - 1) / Config::kCudaTileSize;
    const dim3 grid(num_tiles);
    const dim3 block(Config::kCudaBlockSize);

    unsigned long seed = std::chrono::system_clock::now().time_since_epoch().count();

    nodeid_t *tmp_src;
    nodeid_t *tmp_dst;
    CUDA_CALL(cudaMalloc(&tmp_src, sizeof(nodeid_t) * num_input * fanout));
    CUDA_CALL(cudaMalloc(&tmp_dst, sizeof(nodeid_t) * num_input * fanout));

    sample<Config::kCudaBlockSize, Config::kCudaTileSize>
        <<<grid, block, 0, stream>>>(indptr, indices, num_node, num_edge, input,
                                     num_input, fanout, tmp_src, tmp_dst, num_out, seed);
    CUDA_CALL(cudaGetLastError());

    nodeid_t *item_prefix;
    CUDA_CALL(cudaMalloc(&item_prefix, sizeof(nodeid_t) * (num_input + 1)));
    
    count_edge<Config::kCudaBlockSize, Config::kCudaTileSize>
        <<<grid, block, 0, stream>>>(tmp_src, item_prefix, num_input, fanout);
    CUDA_CALL(cudaGetLastError());

    size_t workspace_bytes;
    CUDA_CALL(cub::DeviceScan::ExclusiveSum(
        nullptr, workspace_bytes, static_cast<nodeid_t *>(nullptr),
        static_cast<nodeid_t *>(nullptr), grid.x + 1));
        
    void *workspace;
    CUDA_CALL(cudaMalloc(&workspace, workspace_bytes));
    CUDA_CALL(cub::DeviceScan::ExclusiveSum(
        workspace, workspace_bytes, item_prefix, item_prefix, grid.x + 1));

    CUDA_CALL(cudaFree(workspace));

    compact_edge<Config::kCudaBlockSize, Config::kCudaTileSize>
        <<<grid, block, 0, stream>>>(tmp_src, tmp_dst, out_src, out_dst, num_out,
                                     item_prefix, num_input, fanout);
    CUDA_CALL(cudaGetLastError());

    CUDA_CALL(cudaFree(tmp_src));
    CUDA_CALL(cudaFree(tmp_dst));
}

} // namespace cuda
} // namespace common
} // namespace samgraph