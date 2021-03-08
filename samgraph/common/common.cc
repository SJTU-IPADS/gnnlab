#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

#include <numeric>
#include <cstdlib>

#include <cuda_runtime.h> 

#include "common.h"
#include "logging.h"

namespace samgraph {
namespace common {

DataContainer::~DataContainer() {
    if (_data == nullptr) {
        return;
    }
        
    switch(_device) {
        case CPU_DEVICE_ID:
            free(_data);
            return;
        case CPU_DEVICE_MMAP_ID:
            munmap(_data, _size);
            return;
        default:
            CUDA_CALL(cudaFree(_data));
            return;
    }
}

bool DataContainer::Consume() {
    if (_data) {
        _data = nullptr;
        return true;
    } else {
        return false;
    }
}

std::shared_ptr<Tensor> Tensor::FromMmap(std::string filepath, DataType dtype, 
                                 std::vector<size_t> dims, int device) {
    auto tensor = std::make_shared<Tensor>();
    size_t expected_size = std::accumulate(dims.begin(), dims.end(), 0) * getDataTypeLength(dtype);
    
    struct stat st;
    stat(filepath.c_str(), &st);
    size_t size = st.st_size;
    SAM_CHECK_EQ(size, expected_size);

    int fd = open(filepath.c_str(), O_RDONLY, 0);
    void *data = mmap(NULL, size, PROT_READ, MAP_SHARED|MAP_FILE, fd, 0);
    mlock(data, size);
    close(fd);

    // if the device is cuda, we have to copy the data from host memory to cuda memory
    if (device > CPU_DEVICE_ID) {
        void *d_data;
        CUDA_CALL(cudaSetDevice(device));
        CUDA_CALL(cudaMalloc(&d_data, size));
        CUDA_CALL(cudaMemcpy(d_data, data, size, cudaMemcpyHostToDevice));
        munmap(data, size);
        data = d_data;
    } else if (device != CPU_DEVICE_MMAP_ID){
        SAM_CHECK(0) << "Invalid device ID";
    }

    tensor->_dtype = dtype;
    tensor->_size = size;
    tensor->_dims = dims;
    tensor->_offset = 0;
    tensor->_container = std::make_shared<DataContainer>(data, size, device);

    return tensor;
}

std::shared_ptr<Tensor> Tensor::Empty(DataType dtype, std::vector<size_t> dims, int device) {
    auto tensor = std::make_shared<Tensor>();
    size_t size = std::accumulate(dims.begin(), dims.end(), 0) * getDataTypeLength(dtype);
    
    void *data;
    if (device > CPU_DEVICE_ID) {
        CUDA_CALL(cudaSetDevice(device));
        CUDA_CALL(cudaMalloc(&data, size));
    } else if (device == CPU_DEVICE_ID) {
        data = malloc(size);
    } else {
        SAM_CHECK(0) << "Unvalid device ID";
    }

    tensor->_dtype = dtype;
    tensor->_dims = dims;
    tensor->_size = size;
    tensor->_offset = 0;
    tensor->_container = std::make_shared<DataContainer>(data, size, device);
    
    return tensor;
}

std::shared_ptr<Tensor> Tensor::CreateView(std::shared_ptr<Tensor> src, size_t offset, std::vector<size_t> dims) {
    SAM_CHECK(src && src->defined());
    auto tensor = std::make_shared<Tensor>();

    tensor->_dtype = src->_dtype;
    tensor->_dims = dims;
    tensor->_size = std::accumulate(dims.begin(), dims.end(), 0) * getDataTypeLength(src->_dtype);
    tensor->_offset = offset + src->_offset;
    tensor->_container = src->_container;

    SAM_CHECK_LE(offset + tensor->_size, src->_size);

    return tensor;
}

std::shared_ptr<Tensor> Tensor::DeepCopy(std::shared_ptr<Tensor> src, size_t offset, std::vector<size_t> dims) {
    SAM_CHECK(src && src->defined());
    auto tensor = std::make_shared<Tensor>();
    size_t size = std::accumulate(dims.begin(), dims.end(), 0) * getDataTypeLength(src ->_dtype);

    tensor->_dtype = src->_dtype;
    tensor->_dims = dims;
    tensor->_size = size;
    tensor->_offset = offset;

    SAM_CHECK_LE(offset + size, src->_size);

    auto src_container = src->_container;
    auto device = src_container->_device;
    auto src_data = src_container->_data;
    void *data;
    if (src_container->_device > CPU_DEVICE_ID) {
        CUDA_CALL(cudaSetDevice(device));
        CUDA_CALL(cudaMalloc(&data, size));
        CUDA_CALL(cudaMemcpy(data, src_data, size, cudaMemcpyDeviceToDevice));
    } else {
        data = malloc(tensor->_size);
        memcpy(data, src_data, size);
        device = CPU_DEVICE_ID;
    }

    tensor->_container = std::make_shared<DataContainer>(data, size, device);

    return tensor;
}

std::shared_ptr<Tensor> Tensor::FromBlob(const void * const data, DataType dtype, std::vector<size_t> dims, int device) {
    auto tensor = std::make_shared<Tensor>();
    size_t size = std::accumulate(dims.begin(), dims.end(), 0) * getDataTypeLength(dtype);
    
    tensor->_dtype = dtype;
    tensor->_dims = dims;
    tensor->_size = size;
    tensor->_offset = 0;
    tensor->_container = std::make_shared<DataContainer>(data, size, device);

    return tensor;
}

uint64_t encodeKey(int epoch_idx, size_t batch_idx) {
    return (epoch_idx << 31) + batch_idx;
}

int getDataTypeLength(int dtype) {
  switch (dtype) {
    case kSamI8:
    case kSamU8:
      return 1;
    case kSamF16:
      return 2;
    case kSamF32:
    case kSamI32:
      return 4;
    case kSamI64:
    case kSamF64:
      return 8;
    default:
      SAM_CHECK(0) << "Unsupported data type: " << dtype;
  }
  return 4;
}


} // namespace common
} // namespace samgraph
