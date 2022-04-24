#include <bits/stdc++.h>
#include <curand.h>
#include <curand_kernel.h>
#include "utility.h"
using namespace std;

// __global__ void simulate_sampling(
//     const int* nodes, const int node_size,
//     const int* edges, const int edge_size,
//     const int
// ) {

// }

enum class SampleType {
    VertexParallel,
    SampleParallel
};

namespace std_sample {

using IdType = uint32_t;

template <size_t TILE_SIZE>
__global__ void vertex_parallel_khop0(
    const IdType *indptr, const IdType *indices,
    const IdType *input, const size_t num_input,
    const size_t fanout, 
    IdType *tmp_src, IdType *tmp_dst
) {
    // assert(WARP_SIZE == blockDim.x);
    // assert(BLOCK_WARP == blockDim.y);
    size_t index = TILE_SIZE * blockIdx.x + threadIdx.y;
    const size_t last_index = min(TILE_SIZE * (blockIdx.x + 1), num_input);

    size_t i =  blockIdx.x * blockDim.x * blockDim.y + threadIdx.x * blockDim.y + threadIdx.y;
    // i is out of bound in num_random_states, so use a new curand
    curandState local_state;
    curand_init(i, 0, 0, &local_state);

    while (index < last_index) {
        const IdType rid = input[index];
        const IdType off = indptr[rid];
        const IdType len = indptr[rid + 1] - indptr[rid];

        if (len <= fanout) {
        size_t j = threadIdx.x;
        for (; j < len; j += blockDim.x) {
            tmp_src[index * fanout + j] = rid;
            tmp_dst[index * fanout + j] = indices[off + j];
        }
        __syncwarp();
        for (; j < fanout; j += blockDim.x) {
                tmp_src[index * fanout + j] = -1;
                tmp_dst[index * fanout + j] = -1;
            }
        } else {
            size_t j = threadIdx.x;
            for (; j < fanout; j += blockDim.x) {
                tmp_src[index * fanout + j] = rid;
                tmp_dst[index * fanout + j] = indices[off + j];
        }
        __syncwarp();
        for (; j < len; j += blockDim.x) {
                size_t k = curand(&local_state) % (j + 1);
                if (k < fanout) {
                atomicExch(tmp_dst + index * fanout + k, indices[off + j]);
                }
            }
        }
        index += blockDim.y;
    }
}

template <size_t TILE_SIZE>
__global__ void sample_parallel_khop0(
    const IdType *indptr, const IdType *indices,
    const IdType *input, const size_t num_input,
    const size_t fanout, 
    IdType *tmp_src, IdType *tmp_dst
) {
    const size_t block_start = TILE_SIZE * blockIdx.x;
    const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    // curandState local_state = random_states[i];
    curandState_t local_state;
    curand_init(i, 0, 0, &local_state);
    for (size_t index = threadIdx.x + block_start; index < block_end;
        index += blockDim.x) {
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
                    tmp_src[index * fanout + j] = -1;
                    tmp_dst[index * fanout + j] = -1;
                }
            } else {
                for (size_t j = 0; j < fanout; ++j) {
                    tmp_src[index * fanout + j] = rid;
                    tmp_dst[index * fanout + j] = indices[off + j];
                }
                for (size_t j = fanout; j < len; ++j) {
                    size_t k = curand(&local_state) % (j + 1);
                    if (k < fanout) {
                        tmp_dst[index * fanout + k] = indices[off + j];
                    }
                }
            }
        }
    }
}

}

float Test(Dataset& dataset, size_t input_num, size_t fanout, int device, SampleType sample_type, int repeat = 5) {
    cudaStream_t stream;
    vector<cudaEvent_t> start(repeat), end(repeat);
    volatile int* start_flag;
    CUDA_CALL(cudaSetDevice(device));
    CUDA_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    for(int i = 0; i < repeat; i++) {
        CUDA_CALL(cudaEventCreate(&start[i]));
        CUDA_CALL(cudaEventCreate(&end[i]));
    }
    CUDA_CALL(cudaHostAlloc(&start_flag, sizeof(int), cudaHostAllocPortable));

    uint32_t *input, *out_src, *out_dst;
    CUDA_CALL(cudaMalloc(&input, sizeof(uint32_t) * input_num));
    CUDA_CALL(cudaMalloc(&out_src, sizeof(uint32_t) * input_num * fanout));
    CUDA_CALL(cudaMalloc(&out_dst, sizeof(uint32_t) * input_num * fanout));

    float tot_ms = 0;
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dist(0, input_num - 1);
    auto cpu_input = make_unique<uint32_t[]>(input_num);
    for(int r = 0; r < repeat; r++) {
        for(int i = 0; i < input_num; i++) {
            cpu_input[i] = dist(gen);
        }
        CUDA_CALL(cudaMemcpy(input, cpu_input.get(), sizeof(uint32_t) * input_num, cudaMemcpyDefault));

        const int WARP_SIZE = 32;
        const int BLOCK_WARP = 128 / WARP_SIZE;
        const int TILE_SIZE = BLOCK_WARP * 16;
        const dim3 block_t(WARP_SIZE, BLOCK_WARP);
        const dim3 grid_t((input_num + TILE_SIZE - 1) / TILE_SIZE);
        
        *start_flag = 0;
        CUDA_CALL(cudaStreamSynchronize(stream));

        delay<<<1, 1, 0, stream>>>(start_flag);
        CUDA_CALL(cudaEventRecord(start[r], stream));
        if (sample_type == SampleType::VertexParallel) {
            std_sample::vertex_parallel_khop0<TILE_SIZE><<<grid_t, block_t, 0, stream>>>(
                dataset.indptr, dataset.indices, input, input_num, fanout, out_src, out_dst);
        } else if (sample_type == SampleType::SampleParallel) {
            std_sample::sample_parallel_khop0<TILE_SIZE><<<grid_t, block_t, 0, stream>>>(
                dataset.indptr, dataset.indices, input, input_num, fanout, out_src, out_dst);
        }
        CUDA_CALL(cudaEventRecord(end[r], stream));
        
        *start_flag = 1;
        CUDA_CALL(cudaStreamSynchronize(stream));
        float ms;
        CUDA_CALL(cudaEventElapsedTime(&ms, start[r], end[r]));
        tot_ms += ms;
    }

    CUDA_CALL(cudaFree(input));;
    CUDA_CALL(cudaFree(out_src));
    CUDA_CALL(cudaFree(out_dst));
    CUDA_CALL(cudaFreeHost((void*)start_flag));
    CUDA_CALL(cudaStreamDestroy(stream));
    for(int i = 0; i < repeat; i++) {
        CUDA_CALL(cudaEventDestroy(start[i]));
        CUDA_CALL(cudaEventDestroy(end[i]));
    }
    return tot_ms / repeat;
}

int main() {
    Dataset dataset("papers100M");

    dataset.hostAllocMapped(0);
    // cout << Test(dataset, 8000, 5, 0, SampleType::SampleParallel, 20) << "\n";
    cout << Test(dataset, 8000, 15, 0, SampleType::VertexParallel, 10) << "\n";
    dataset.p2p(0, 1);
    // cout << Test(dataset, 8000, 5, 0, SampleType::SampleParallel, 20) << "\n";
    cout << Test(dataset, 8000, 15, 0, SampleType::VertexParallel, 10) << "\n";
    // cout << Test(dataset, 100000, 5, 0) << "\n";
}