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
    SampleParallel,
    KHop2,
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
    // if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0 ) {
    //     printf("input: ");
    //     for(int i = 0; i < 10; i++) {
    //         printf("%d ", input[i]);
    //     }
    //     printf("\n");
    //     printf("indptr: ");
    //     for(int i = 0; i < 10; i++) {
    //         printf("%d ", indptr[i]);
    //     }
    //     printf("\n");
    //     printf("indices: ");
    //     for(int i = 0; i < 10; i++) {
    //         printf("%d ", indices[i]);
    //     }
    // }
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
    // if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0 ) {
    //     printf("output: ");
    //     for(int i = num_input; i < num_input + 20; i++) {
    //         printf("(%d %d)", tmp_src[i], tmp_dst[i]);
    //     }
    //     printf("\n");
    // }
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

template <size_t TILE_SIZE>
__global__ void sample_khop2(const IdType *indptr, IdType *indices,
                             const IdType *input, const size_t num_input,
                             const size_t fanout, IdType *tmp_src,
                             IdType *tmp_dst
                            //  , curandState *random_states,
                            //  size_t num_random_states
                             ) {
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
//   assert(i < num_random_states);
//   curandState local_state = random_states[i];
  curandState local_state;
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
          // now we have $len - j$ cancidate
          size_t selected_j = curand(&local_state) % (len - j);
          IdType selected_node_id = indices[off + selected_j];
          tmp_src[index * fanout + j] = rid;
          tmp_dst[index * fanout + j] = selected_node_id;
          // swap indices[off+len-j-1] with indices[off+ selected_j];
          indices[off + selected_j] = indices[off+len-j-1];
          indices[off+len-j-1] = selected_node_id;
        }
      }
    }
  }

//   random_states[i] = local_state;
}

}



enum class InputType {
    RandomTrainset,
    SequentialTrainset,
    RandomVertex,
    SequantialVertex,
    FromFile,
};

void statistic_sample_input(Dataset& dataset, uint32_t* input, size_t input_num) {
    // auto indptr = make_unique<uint32_t[]>(dataset.node_num + 1);
    // auto indices = make_unique<uint32_t[]>(dataset.edge_num);
    auto graph = dataset.get_cpu_graph();
    auto indptr = graph.first.get();
    auto indices = graph.second.get();
    map<string, double> res;
    res["sample input_num"] = input_num;
    
    vector<size_t> node_pos;
    vector<size_t> edge_pos;
    vector<size_t> degree;
    for(size_t i = 0; i < input_num; i++) {
        uint32_t v = input[i];
        assert(v < dataset.node_num);
        uint32_t off = indptr[v];
        uint32_t len = indptr[v + 1] - indptr[v];
        node_pos.push_back(v);
        edge_pos.push_back(off);
        degree.push_back(len);
    }
    auto degree_sta = sum_avg_std(degree);
    auto edge_pos_sta = sum_avg_std(edge_pos);
    auto node_pos_sta = sum_avg_std(node_pos);
    res["sum(edge_num)"] = get<0>(degree_sta);
    res["avg(edge_num)"] = get<1>(degree_sta);
    res["std(edge_num)"] = get<2>(degree_sta);
    res["avg(node_pos)"] = get<1>(node_pos_sta);
    res["std(node_pos)"] = get<2>(node_pos_sta);
    res["std(edge_pos)"] = get<2>(edge_pos_sta);

    size_t fw1 = 0, fw2 = 0;
    for(auto v : res) {
        fw1 = max(fw1, v.first.size());
        fw2 = max(fw2, std::to_string(v.second).size());
    }
    cout << setw(fw1 + fw2 + 8) << setfill('-') << "-";
    cout << "\n";
    cout.fill(' ');
    for(auto v : res) {
        cout << "| " << left << setw(fw1) << v.first << " | ";
        cout << left << setw(fw2) << to_string(v.second) << " |\n";
    }
    cout << setw(fw1 + fw2 + 8) << setfill('-') << "-";
    cout << "\n";
}

// if input from file, input_num will be ignored and will be set according to input file
// for other input type, input_file will be ignored
float Test(Dataset& dataset, InputType input_type ,size_t input_num, size_t fanout, int device, 
    SampleType sample_type, string input_file, int repeat = 5, bool verbose = false) {
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
    // auto cpu_input = make_unique<uint32_t[]>(input_num);
    unique_ptr<uint32_t[]> cpu_input;
    if (input_type == InputType::FromFile) {
        ifstream ifs(input_file, ios::binary);
        if(!ifs) { 
            printf("input file %s dose not exist\n", input_file.c_str()); 
            exit(EXIT_FAILURE); 
        }
        ifs.seekg(0, ios::end);
        input_num = ifs.tellg() / sizeof(uint32_t);
        // printf("input-fanout-5 num: %d", input_num);
        ifs.seekg(0);
        cpu_input = make_unique<uint32_t[]>(input_num);
        ifs.read((char*)cpu_input.get(), sizeof(uint32_t) * input_num);
        // CUDA_CALL(cudaMalloc(&input, sizeof(uint32_t) * input_num));
        // CUDA_CALL(cudaMalloc(&out_src, sizeof(uint32_t) * input_num * fanout));
        // CUDA_CALL(cudaMalloc(&out_dst, sizeof(uint32_t) * input_num * fanout));
    } else {
        cpu_input = make_unique<uint32_t[]>(input_num);
    }
    CUDA_CALL(cudaMalloc(&input, sizeof(uint32_t) * input_num));
    CUDA_CALL(cudaMalloc(&out_src, sizeof(uint32_t) * input_num * fanout));
    CUDA_CALL(cudaMalloc(&out_dst, sizeof(uint32_t) * input_num * fanout));

    float tot_ms = 0;
    random_device rd;
    mt19937 gen(rd());
    int trainset_input_idx = 0;
    int r;
    for(r = 0; r < repeat; r++) {
        switch (input_type)
        {
        case InputType::RandomTrainset: {
            uniform_int_distribution<> dist(0, dataset.train_num - 1);
            for(int i = 0; i < input_num; i++) {
                cpu_input[i] = dist(gen);
            }   
        }
            break;
        case InputType::SequentialTrainset:
            input_num = min(input_num, dataset.train_num - trainset_input_idx);
            for(int i = 0; i < input_num; i++, trainset_input_idx++)
                cpu_input[i] = dataset.trainset[trainset_input_idx];
            break;
        case InputType::RandomVertex: {
            uniform_int_distribution<> dist(0, dataset.node_num - 1);
            for(int i = 0; i < input_num; i++) {
                cpu_input[i] = dist(gen);
            }
        }
            break;
        case InputType::SequantialVertex:
            input_num = min(input_num, dataset.node_num - trainset_input_idx);
            for(int i = 0; i < input_num; i++, trainset_input_idx++)
                cpu_input[i] = trainset_input_idx;
            break;
        case InputType::FromFile:
            break;
        default:
            assert(false);
        }

        if (input_num == 0) {
            break;
        }
        if(verbose) {
            printf("[Test repeat=%d] input num=%lld fanout=%lld\n", r, input_num, fanout);
            statistic_sample_input(dataset, cpu_input.get(), input_num);
        }
        CUDA_CALL(cudaSetDevice(device));
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
        } else if (sample_type == SampleType::KHop2) {
            // const size_t num_tiles = RoundUpDiv(input_num, Constant::kCudaTileSize);
            const size_t kCudaTileSize = 1024;
            const size_t kCudaBlockSize = 256;
            const size_t num_tiles = (input_num + kCudaTileSize - 1) / kCudaTileSize;
            const dim3 grid(num_tiles);
            const dim3 block(kCudaBlockSize);
            std_sample::sample_khop2<TILE_SIZE> <<<grid, block, 0, stream>>> (
                dataset.indptr, dataset.indices, input, input_num, fanout, out_src, out_dst);
        } else {
            assert(false);
        }
        CUDA_CALL(cudaEventRecord(end[r], stream));
        
        *start_flag = 1;
        CUDA_CALL(cudaStreamSynchronize(stream));
        float ms;
        CUDA_CALL(cudaEventElapsedTime(&ms, start[r], end[r]));
        tot_ms += ms;
        if (verbose) {
            printf("[Test repeat=%d] %f\n", r, ms);
        }
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
    return tot_ms / r;
}

int main() {
    // Dataset dataset("papers100M");
    // Dataset dataset("uk-2006-05");
    Dataset dataset("com-friendster");
    float sample_time;

    int local_device = 0;
    int remote_device = 1;

    dataset.p2p(local_device, remote_device);
    {
        // sample_time = Test(dataset, InputType::FromFile,0, 5, local_device, 
        //     SampleType::VertexParallel, "/disk1/wjl/samgraph/input-fanout-5.bin", 1, true);
        // printf("p2p FromFile %f\n\n", sample_time);

        // sample_time = Test(dataset, InputType::SequentialTrainset, 500000, 5, local_device,
        //     SampleType::VertexParallel, "", 1, true);
        // printf("p2p SequentialTrainset %f\n\n", sample_time);

        // sample_time = Test(dataset, InputType::RandomTrainset, 500000, 5, local_device, 
        //     SampleType::VertexParallel, "", 2, true);
        // printf("p2p RandomTrainset %f\n\n", sample_time);

        // sample_time = Test(dataset, InputType::SequantialVertex, 500000, 5, local_device,
        //     SampleType::VertexParallel, "", 1, true);
        // printf("p2p SequantialVertex %f\n\n", sample_time);

        sample_time = Test(dataset, InputType::RandomVertex, 20000, 5, local_device,
            SampleType::VertexParallel, "", 2, true );
        printf("p2p RandomVertex %f\n\n", sample_time);
    }

    // dataset.hostAllocMapped(local_device);
    // {
    //     // sample_time = Test(dataset, InputType::FromFile,0, 5, 0, 
    //     //     SampleType::VertexParallel, "/disk1/wjl/samgraph/input-fanout-5.bin", 1, true);
    //     // printf("mapped FromFile %f\n\n", sample_time);

    //     sample_time = Test(dataset, InputType::SequentialTrainset, 500000, 5, local_device,
    //         SampleType::VertexParallel, "", 1, true);
    //     printf("mapped SequentialTrainset %f\n\n", sample_time);

    //     sample_time = Test(dataset, InputType::RandomTrainset, 500000, 5, local_device, 
    //         SampleType::VertexParallel, "", 2, true);
    //     printf("mapped RandomTrainset %f\n\n", sample_time);

    //     sample_time = Test(dataset, InputType::SequantialVertex, 500000, 5, local_device,
    //         SampleType::VertexParallel, "", 1, true);
    //     printf("mapped SequantialVertex %f\n\n", sample_time);

    //     sample_time = Test(dataset, InputType::RandomVertex, 500000, 5, local_device,
    //         SampleType::VertexParallel, "", 2, true );
    //     printf("mapped RandomVertex %f\n\n", sample_time);
    // }
}