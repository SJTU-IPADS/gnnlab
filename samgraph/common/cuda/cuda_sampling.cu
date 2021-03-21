#include <chrono>
#include <cassert>
#include <cstdio>

#include <curand.h>
#include <curand_kernel.h>
#include <cub/cub.cuh>

#include "../config.h"
#include "../logging.h"
#include "../common.h"
#include "cuda_function.h"

namespace {

template <typename T>
struct BlockPrefixCallbackOp {
    T _running_total;
    
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

template<size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void
sample(const IdType *indptr, const IdType *indices, const IdType *input, const size_t num_input,
       const size_t fanout, IdType *tmp_src, IdType *tmp_dst, unsigned long seed) {
    assert(BLOCK_SIZE == blockDim.x);
    const size_t block_start = TILE_SIZE * blockIdx.x;
    const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    curandState state;
    curand_init(seed, i, 0, &state);

    #pragma unroll
    for (size_t index = threadIdx.x + block_start; index < block_end; index += BLOCK_SIZE) {
        if (index < num_input) {
            const IdType rid = input[index];
            const IdType off = indptr[rid];
            const IdType len = indptr[rid + 1] - indptr[rid];

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

template<size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void
count_edge(IdType *edge_src, size_t *item_prefix, const size_t num_input, const size_t fanout) {
    assert(BLOCK_SIZE == blockDim.x);
    
    using BlockReduce = typename cub::BlockReduce<size_t, BLOCK_SIZE>;
    
    const size_t block_start = TILE_SIZE * blockIdx.x;
    const size_t block_end = TILE_SIZE * (blockIdx.x + 1);
    
    size_t count = 0;
    #pragma unroll
    for (size_t index = threadIdx.x + block_start; index < block_end; index += BLOCK_SIZE) {
        if (index < num_input) {
            for (size_t j = 0; j < fanout; j++) {
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

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void 
compact_edge(const IdType *tmp_src, const IdType *tmp_dst, IdType *out_src, IdType *out_dst,
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
                if (tmp_src[index * fanout + j] != Config::kEmptyKey) {
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

void DeviceSample(const IdType *indptr, const IdType *indices, const IdType *input, const size_t num_input,
                  const size_t fanout, IdType *out_src, IdType *out_dst, size_t *num_out, cudaStream_t stream) {
    SAM_LOG(DEBUG) << "DeviceSample: begin with num_input " << num_input << " and fanout " << fanout;

    const size_t num_tiles = (num_input + Config::kCudaTileSize - 1) / Config::kCudaTileSize;
    const dim3 grid(num_tiles);
    const dim3 block(Config::kCudaBlockSize);

    unsigned long seed = std::chrono::system_clock::now().time_since_epoch().count();
    seed = 10ul;

    IdType *tmp_src;
    IdType *tmp_dst;
    CUDA_CALL(cudaMalloc(&tmp_src, sizeof(IdType) * num_input * fanout));
    CUDA_CALL(cudaMalloc(&tmp_dst, sizeof(IdType) * num_input * fanout));
    SAM_LOG(DEBUG) << "DeviceSample: cuda tmp_src malloc " << toReadableSize(num_input * fanout * sizeof(IdType));
    SAM_LOG(DEBUG) << "DeviceSample: cuda tmp_dst malloc " << toReadableSize(num_input * fanout * sizeof(IdType));

    sample<Config::kCudaBlockSize, Config::kCudaTileSize>
        <<<grid, block, 0, stream>>>(indptr, indices, input, num_input,
                                     fanout, tmp_src, tmp_dst, seed);
    CUDA_CALL(cudaStreamSynchronize(stream));

    size_t *item_prefix;
    CUDA_CALL(cudaMalloc(&item_prefix, sizeof(size_t) * (num_input + 1)));
    SAM_LOG(DEBUG) << "DeviceSample: cuda item_prefix malloc " << toReadableSize(sizeof(size_t) * (num_input + 1));
    
    count_edge<Config::kCudaBlockSize, Config::kCudaTileSize>
        <<<grid, block, 0, stream>>>(tmp_src, item_prefix, num_input, fanout);
    CUDA_CALL(cudaStreamSynchronize(stream));

    size_t workspace_bytes;
    CUDA_CALL(cub::DeviceScan::ExclusiveSum(
        nullptr, workspace_bytes, static_cast<size_t *>(nullptr),
        static_cast<size_t *>(nullptr), grid.x + 1, stream));
    CUDA_CALL(cudaStreamSynchronize(stream));

    void *workspace;
    CUDA_CALL(cudaMalloc(&workspace, workspace_bytes));
    CUDA_CALL(cub::DeviceScan::ExclusiveSum(
        workspace, workspace_bytes, item_prefix, item_prefix, grid.x + 1, stream));
    CUDA_CALL(cudaStreamSynchronize(stream));
    SAM_LOG(DEBUG) << "DeviceSample: cuda workspace malloc " << toReadableSize(workspace_bytes);

    compact_edge<Config::kCudaBlockSize, Config::kCudaTileSize>
        <<<grid, block, 0, stream>>>(tmp_src, tmp_dst, out_src, out_dst, num_out,
                                     item_prefix, num_input, fanout);
    CUDA_CALL(cudaStreamSynchronize(stream));

    CUDA_CALL(cudaFree(workspace));
    CUDA_CALL(cudaFree(item_prefix));
    CUDA_CALL(cudaFree(tmp_src));
    CUDA_CALL(cudaFree(tmp_dst));
    SAM_LOG(DEBUG) << "DeviceSample: succeed ";
}

} // namespace cuda
} // namespace common
} // namespace samgraph