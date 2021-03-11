#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

#include <cstring>
#include <numeric>
#include <cstdlib>

#include <cuda_runtime.h> 

#include "common.h"
#include "logging.h"
#include "config.h"

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

void DataContainer::Consume() {
    SAM_CHECK_GE(_device, CPU_DEVICE_ID);
    SAM_CHECK(!_is_consumed);
    _is_consumed = true;
    _data = nullptr;
}

std::shared_ptr<Tensor> Tensor::FromMmap(std::string filepath, DataType dtype, 
                                         std::vector<size_t> dims, int device) {
    auto tensor = std::make_shared<Tensor>();
    size_t expected_size = std::accumulate(dims.begin(), dims.end(), 1ul, std::multiplies<size_t>()) * getDataTypeLength(dtype);
    
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
        SAM_LOG(DEBUG) << "FromMmap: cuda malloc " << toReadableSize(size);
        CUDA_CALL(cudaMemcpy(d_data, data, size, cudaMemcpyHostToDevice));
        munmap(data, size);
        data = d_data;
    } else if (device == CPU_DEVICE_ID){
        SAM_LOG(DEBUG) << "FromMmap: cpu malloc " << toReadableSize(size);
        void *new_data = malloc(size);
        memcpy(new_data, data, size);
        munmap(data, size);
        data = new_data;
    } else {
        SAM_LOG(DEBUG) << "FromMmap: mmap " << toReadableSize(size);
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
    size_t size = std::accumulate(dims.begin(), dims.end(), 1ul, std::multiplies<size_t>()) * getDataTypeLength(dtype);
    
    void *data;
    if (device > CPU_DEVICE_ID) {
        CUDA_CALL(cudaSetDevice(device));
        CUDA_CALL(cudaMalloc(&data, size));
        SAM_LOG(DEBUG) << "Empty: cuda malloc " << toReadableSize(size);
    } else if (device == CPU_DEVICE_ID) {
        data = malloc(size);
        SAM_LOG(DEBUG) << "Empty: cpu malloc " << toReadableSize(size);
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
    tensor->_size = std::accumulate(dims.begin(), dims.end(), 1ul, std::multiplies<size_t>()) * getDataTypeLength(src->_dtype);
    tensor->_offset = offset + src->_offset;
    tensor->_container = src->_container;

    SAM_CHECK_LE(offset + tensor->_size, src->_size);

    return tensor;
}

std::shared_ptr<Tensor> Tensor::DeepCopy(std::shared_ptr<Tensor> src, size_t offset, std::vector<size_t> dims) {
    SAM_CHECK(src && src->defined());
    auto tensor = std::make_shared<Tensor>();
    size_t size = std::accumulate(dims.begin(), dims.end(), 1ul, std::multiplies<size_t>()) * getDataTypeLength(src ->_dtype);

    tensor->_dtype = src->_dtype;
    tensor->_dims = dims;
    tensor->_size = size;
    tensor->_offset = offset;

    SAM_CHECK_LE(offset + size, src->_size);

    auto src_container = src->_container;
    auto device = src->device();
    auto src_data = src->data();
    void *data;
    if (device > CPU_DEVICE_ID) {
        CUDA_CALL(cudaSetDevice(device));
        CUDA_CALL(cudaMalloc(&data, size));
        CUDA_CALL(cudaMemcpy(data, src_data, size, cudaMemcpyDeviceToDevice));
        SAM_LOG(DEBUG) << "DeepCopy: cuda malloc " << toReadableSize(size);
    } else {
        data = malloc(tensor->_size);
        memcpy(data, src_data, size);
        device = CPU_DEVICE_ID;
        SAM_LOG(DEBUG) << "DeepCopy: cpu malloc " << toReadableSize(size);
    }

    tensor->_container = std::make_shared<DataContainer>(data, size, device);

    return tensor;
}

std::shared_ptr<Tensor> Tensor::FromBlob(void *data, DataType dtype, std::vector<size_t> dims, int device) {
    auto tensor = std::make_shared<Tensor>();
    size_t size = std::accumulate(dims.begin(), dims.end(), 1ul, std::multiplies<size_t>()) * getDataTypeLength(dtype);
    
    tensor->_dtype = dtype;
    tensor->_dims = dims;
    tensor->_size = size;
    tensor->_offset = 0;
    tensor->_container = std::make_shared<DataContainer>(data, size, device);

    return tensor;
}

uint64_t encodeBatchKey(int epoch_idx, size_t batch_idx) {
    return (epoch_idx << 31) + (batch_idx << 15);
}

uint64_t encodeGraphID(uint64_t key, int layer_idx) {
    return (key & Config::kBatchMask) + layer_idx;
}

int decodeGraphID(uint64_t key) {
    return key & 0xffff;
}

size_t getDataTypeLength(int dtype) {
  switch (dtype) {
    case kSamI8:
    case kSamU8:
      return 1ul;
    case kSamF16:
      return 2ul;
    case kSamF32:
    case kSamI32:
      return 4ul;
    case kSamI64:
    case kSamF64:
      return 8ul;
    default:
      SAM_CHECK(0) << "Unsupported data type: " << dtype;
  }
  return 4ul;
}

void cudaDataDeleter(void *data) {
    CUDA_CALL(cudaFree(data));
}

void cpuDataDeleter(void *data) {
    free(data);
}

std::string toReadableSize(size_t size_in_bytes) {
    char buf[Config::kBufferSize];
    if (size_in_bytes > Config::kGigabytes) {
        double new_size = (float) size_in_bytes / Config::kGigabytes;
        sprintf(buf, "%.2lf GB", new_size);
        return std::string(buf);
    } else if (size_in_bytes > Config::kMegabytes) {
        double new_size = (float) size_in_bytes / Config::kMegabytes;
        sprintf(buf, "%.2lf MB", new_size);
        return std::string(buf);
    } else if (size_in_bytes > Config::kKilobytes) {
        double new_size = (float) size_in_bytes / Config::kKilobytes;
        sprintf(buf, "%.2lf KB", new_size);
        return std::string(buf);
    } else {
        double new_size = (float) size_in_bytes;
        sprintf(buf, "%.2lf Bytes", new_size);
        return std::string(buf);
    }
}


} // namespace common
} // namespace samgraph
