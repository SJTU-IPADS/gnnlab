#include <fstream>
#include <cassert>
#include <cstring>
#include <random>
#include <curand.h>
#include <curand_kernel.h>
#include "utility.h"

__global__ void delay(volatile int* flag) {
    while(!*flag) {
    }
} 

__global__ void read(int* __restrict__ arr, int len, int* __restrict__ result, int result_len) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t grid_size = blockDim.x * gridDim.x;
    size_t rid = idx % result_len;
#pragma unroll(5)
    for(size_t i = idx; i < len; i += grid_size) {
        result[rid] += arr[i];
    }
}

__global__ void random_rand(int* __restrict__ arr, int len, int* __restrict__ result, int result_len, int seed) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t grid_size = blockDim.x * gridDim.x;
    size_t rid = idx % result_len;
    curandState state;
    curand_init(seed + idx, 0, 0, &state);
#pragma unroll(5)
    for(size_t i = 0; i < len; i += grid_size) {
        int x = curand(&state) % len;
        result[rid] += arr[x];
    }
}

__global__ void random_read_overhead(int* __restrict__ arr, int len, int* __restrict__ result, int result_len, int seed) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t grid_size = blockDim.x * gridDim.x;
    size_t rid = idx % result_len;
    curandState state;
    curand_init(seed + idx, 0, 0, &state);
#pragma unroll(5)
    for(size_t i = 0; i < len; i += grid_size) {
        int x = curand(&state) % len;
        result[rid] += x;
    }
}

void perform_sequential_read(
    int grid_size, int block_size, cudaStream_t stream, 
    int* arr, int len, int* result, int result_len
) {
    read<<<grid_size, block_size, 0, stream>>>(arr, len, result, result_len);
}

void perform_random_read_int32(
    int grid_size, int block_size, cudaStream_t stream,
    int* arr, int len, int* result, int result_len
) {
    std::random_device rd;
    std::mt19937 gen(rd());
    uniform_int_distribution<> dist(0, len - 1);
    read<<<1, 1, 0, stream>>>(arr + dist(gen), 1, result, result_len);
}

void perform_random_read(
    int grid_size, int block_size, cudaStream_t stream,
    int* arr, int len, int* result, int result_len
) {
    std::random_device rd;
    std::mt19937 gen(rd());
    random_rand<<<grid_size, block_size, 0, stream>>>(arr, len, result, result_len, gen());
}

void perform_random_read_overhead(
    int grid_size, int block_size, cudaStream_t stream,
    int* arr, int len, int* result, int result_len
) {
    std::random_device rd;
    std::mt19937 gen(rd());
    random_read_overhead<<<grid_size, block_size, 0, stream>>>(arr, len, result, result_len, gen());
}

tuple<double, double, double> sum_avg_std(const vector<size_t> &vec) {
    double sum = 0;
    double var = 0;
    for(auto x : vec) {
        sum += x;
        var += x * x;
    }
    double avg = sum / vec.size();
    var /= vec.size();
    var -= avg * avg;
    assert(var >= 0);
    return {sum, avg, sqrt(var)};
}

string Dataset::root_path = "/graph-learning/samgraph/";

Dataset::Dataset(string name) : name(name), mm_type(MemoryType::CPU), 
    local_device(cudaCpuDeviceId), remote_device(cudaCpuDeviceId) {
    string path = root_path + name + "/";
    ifstream indptr(path + "indptr.bin", ios::binary);
    ifstream indices(path + "indices.bin", ios::binary);
    ifstream trainset(path + "train_set.bin", ios::binary);
    if(!indptr) {
        cout << "file " << path + "indptr.bin" << " not exist\n";
        exit(EXIT_FAILURE);
    }
    if(!indices) {
        cout << "file " << path + "indices.bin" << " not exist\n";
        exit(EXIT_FAILURE);
    }
    if(!trainset) {
        cout << "file " << path + "train_set.bin" << " not exist\n";
    }
    indptr.seekg(0, ios::end);
    size_t file_size = indptr.tellg();
    indptr.seekg(0);
    node_num = file_size / 4 - 1;
    // this->indptr = new uint32_t[file_size / 4];
    CUDA_CALL(cudaHostAlloc(&this->indptr, file_size, cudaHostAllocDefault));
    indptr.read((char*)this->indptr, file_size);
    indptr.close();
    assert(indptr.good());

    indices.seekg(0, ios::end);
    file_size = indices.tellg();
    indices.seekg(0);
    edge_num = file_size / 4;
    // this->indices = new uint32_t[file_size / 4];
    CUDA_CALL(cudaHostAlloc(&this->indices, file_size, cudaHostAllocDefault));
    indices.read((char*)this->indices, file_size);
    indices.close();
    assert(indices.good());

    trainset.seekg(0, ios::end);
    file_size = trainset.tellg();
    trainset.seekg(0);
    train_num = file_size / 4;
    this->trainset = new uint32_t[train_num];
    trainset.read((char*)this->trainset, file_size);
    trainset.close();
    assert(trainset.good());
}

void Dataset::cpu() {
    if (mm_type == MemoryType::CPU)
        return;
    uint32_t *indptr, *indices;
    // indptr = new uint32_t[node_num + 1];
    // indices = new uint32_t[edge_num];
    CUDA_CALL(cudaHostAlloc(&indptr, sizeof(uint32_t) * (node_num + 1), cudaHostAllocDefault));
    CUDA_CALL(cudaHostAlloc(&indices, sizeof(uint32_t) * (edge_num), cudaHostAllocDefault));
    switch (mm_type)
    {
    case MemoryType::HostAllocMapped: 
        CUDA_CALL(cudaMemcpy(indptr, this->indptr, sizeof(uint32_t) * (node_num + 1), cudaMemcpyDefault));
        CUDA_CALL(cudaMemcpy(indices, this->indices, sizeof(uint32_t) * (edge_num), cudaMemcpyDefault));
        // memcpy(indptr, this->indptr, sizeof(uint32_t) * (node_num + 1));
        // memcpy(indices, this->indices, sizeof(uint32_t) * (edge_num));
        break;
    case MemoryType::P2P: 
        CUDA_CALL(cudaSetDevice(remote_device));
        CUDA_CALL(cudaMemcpy(indptr, this->indptr, sizeof(uint32_t) * (node_num + 1), cudaMemcpyDefault));
        CUDA_CALL(cudaMemcpy(indices, this->indices, sizeof(uint32_t) * (edge_num), cudaMemcpyDefault));
        break;
    default :
        assert(false);
    }
    _free_graph();
    this->indptr = indptr;
    this->indices = indices;
    this->mm_type = MemoryType::CPU;
    this->local_device = this->remote_device = cudaCpuDeviceId;
}

void Dataset::p2p(int local_device, int remote_device) {
    if (this->mm_type == MemoryType::P2P) {
        return;
    } else if (mm_type != MemoryType::CPU) {
        cpu();
    }
    this->local_device = local_device;
    this->remote_device = remote_device;
    uint32_t *indptr, *indices;

    CUDA_CALL(cudaSetDevice(remote_device));
    CUDA_CALL(cudaMalloc(&indptr, sizeof(uint32_t) * (node_num + 1)));
    CUDA_CALL(cudaMalloc(&indices, sizeof(uint32_t) * edge_num));
    CUDA_CALL(cudaMemcpy(indptr, this->indptr, sizeof(uint32_t) * (node_num + 1), cudaMemcpyDefault));
    CUDA_CALL(cudaMemcpy(indices, this->indices, sizeof(uint32_t) * (edge_num), cudaMemcpyDefault));
    CUDA_CALL(cudaDeviceSynchronize());
    CUDA_CALL(cudaSetDevice(local_device));
    CUDA_CALL(cudaDeviceEnablePeerAccess(remote_device, 0));

    _free_graph();
    this->indptr = indptr;
    this->indices = indices;
    this->mm_type = MemoryType::P2P;
}

void Dataset::hostAllocMapped(int device) {
    if (this->mm_type == MemoryType::HostAllocMapped) {
        return;
    } else if (mm_type != MemoryType::CPU) {
        cpu();
    }
    this->local_device = this->remote_device = device;
    uint32_t *indptr, *indices;

    CUDA_CALL(cudaHostAlloc(&indptr, sizeof(uint32_t) * (node_num + 1), cudaHostAllocMapped));
    CUDA_CALL(cudaHostAlloc(&indices, sizeof(uint32_t) * edge_num, cudaHostAllocMapped));
    CUDA_CALL(cudaMemcpy(indptr, this->indptr, sizeof(uint32_t) * (node_num + 1), cudaMemcpyDefault));
    CUDA_CALL(cudaMemcpy(indices, this->indices, sizeof(uint32_t) * (edge_num), cudaMemcpyDefault));
    // memcpy(indptr, this->indptr, sizeof(uint32_t) * (node_num + 1));
    // memcpy(indices, this->indices, sizeof(uint32_t) * (edge_num));

    _free_graph();
    this->indptr = indptr;
    this->indices = indices;
    this->mm_type = MemoryType::HostAllocMapped;
}

pair<unique_ptr<uint32_t[]>, unique_ptr<uint32_t[]>> Dataset::get_cpu_graph() {
    auto indptr = make_unique<uint32_t[]>(node_num + 1);
    auto indices = make_unique<uint32_t[]>(edge_num);
    switch (mm_type)
    {
    case MemoryType::CPU:
    case MemoryType::HostAllocMapped:
        CUDA_CALL(cudaMemcpy(indptr.get(), this->indptr, sizeof(uint32_t) * (node_num + 1), cudaMemcpyDefault));
        CUDA_CALL(cudaMemcpy(indices.get(), this->indices, sizeof(uint32_t) * (edge_num), cudaMemcpyDefault));
        break;
    case MemoryType::P2P:
        CUDA_CALL(cudaSetDevice(remote_device));
        CUDA_CALL(cudaMemcpy(indptr.get(), this->indptr, sizeof(uint32_t) * (node_num + 1), cudaMemcpyDefault));
        CUDA_CALL(cudaMemcpy(indices.get(), this->indices, sizeof(uint32_t) * edge_num, cudaMemcpyDefault));
        break;
    case MemoryType::UM_CUDA_CPU:
    case MemoryType::UM_CUDA_CUDA:
    default:
        assert(false);
    }
    return {move(indptr), move(indices)};
}

Dataset::~Dataset() {
    _free_graph();
    delete[] this->trainset;
}

void Dataset::_free_graph() {
    switch (mm_type)
    {
    case MemoryType::CPU:
        // delete[] this->indptr;
        // delete[] this->indices;
        CUDA_CALL(cudaFreeHost(this->indptr));
        CUDA_CALL(cudaFreeHost(this->indices));
        return ;
    case MemoryType::HostAllocMapped:
        CUDA_CALL(cudaFreeHost(this->indptr));
        CUDA_CALL(cudaFreeHost(this->indices));
        return;
    case MemoryType::P2P:
        CUDA_CALL(cudaSetDevice(local_device));
        CUDA_CALL(cudaDeviceDisablePeerAccess(remote_device));
        CUDA_CALL(cudaSetDevice(remote_device));
        CUDA_CALL(cudaFree(this->indptr));
        CUDA_CALL(cudaFree(this->indices));
        return;
    default:
        assert(false);
    }
    
}