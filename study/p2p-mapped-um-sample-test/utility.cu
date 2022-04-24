#include <fstream>
#include <cassert>
#include <cstring>
#include "utility.h"

__global__ void delay(volatile int* flag) {
    while(!*flag) {
    }
} 

__global__ void read(int* arr, int len, int* result, int result_len) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t grid_size = blockDim.x * gridDim.x;
#pragma unroll(5)
    for(size_t i = idx; i < len; i += grid_size) {
        result[i % result_len] += arr[i];
    }
}


string Dataset::root_path = "/graph-learning/samgraph/";

Dataset::Dataset(string name) : name(name), mm_type(MemoryType::CPU), 
    local_device(cudaCpuDeviceId), remote_device(cudaCpuDeviceId) {
    string path = root_path + name + "/";
    ifstream indptr(path + "indptr.bin", ios::binary);
    ifstream indices(path + "indices.bin", ios::binary);
    if(!indptr) {
        cout << "file " << path + "indptr.bin" << " not exist\n";
        exit(EXIT_FAILURE);
    }
    if(!indices) {
        cout << "file " << path + "indices.bin" << " not exist\n";
        exit(EXIT_FAILURE);
    }
    indptr.seekg(0, ios::end);
    size_t file_size = indptr.tellg();
    indptr.seekg(0);
    node_num = file_size / 4 - 1;
    this->indptr = new uint32_t[file_size / 4];
    indptr.read((char*)this->indptr, file_size);
    indptr.close();
    assert(indptr.good());

    indices.seekg(0, ios::end);
    file_size = indices.tellg();
    indices.seekg(0);
    edge_num = file_size / 4;
    this->indices = new uint32_t[file_size / 4];
    indices.read((char*)this->indices, file_size);
    indices.close();
    assert(indices.good());
}

void Dataset::cpu() {
    if (mm_type == MemoryType::CPU)
        return;
    uint32_t *indptr, *indices;
    indptr = new uint32_t[node_num + 1];
    indices = new uint32_t[edge_num];
    switch (mm_type)
    {
    case MemoryType::HostAllocMapped: 
        memcpy(indptr, this->indptr, sizeof(uint32_t) * (node_num + 1));
        memcpy(indices, this->indices, sizeof(uint32_t) * (edge_num));
        break;
    case MemoryType::P2P: 
        CUDA_CALL(cudaSetDevice(remote_device));
        CUDA_CALL(cudaMemcpy(indptr, this->indptr, sizeof(uint32_t) * (node_num + 1), cudaMemcpyDefault));
        CUDA_CALL(cudaMemcpy(indices, this->indices, sizeof(uint32_t) * (edge_num), cudaMemcpyDefault));
        break;
    default :
        assert(false);
    }
    _free();
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
    CUDA_CALL(cudaSetDevice(local_device));
    CUDA_CALL(cudaDeviceEnablePeerAccess(remote_device, 0));

    _free();
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
    memcpy(indptr, this->indptr, sizeof(uint32_t) * (node_num + 1));
    memcpy(indices, this->indices, sizeof(uint32_t) * (edge_num));

    _free();
    this->indptr = indptr;
    this->indices = indices;
    this->mm_type = MemoryType::HostAllocMapped;
}

Dataset::~Dataset() {
    _free();
}

void Dataset::_free() {
    switch (mm_type)
    {
    case MemoryType::CPU:
        delete[] this->indptr;
        delete[] this->indices;
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