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

Tensor::Tensor()
    : _data(nullptr), _device(CPU_DEVICE_ID) {}

Tensor::~Tensor() {
    if (!_data) {
        return;
    }

    SAM_LOG(DEBUG) << "Tensor " << _name << " has been freed";
    switch(_device) {
        case CPU_DEVICE_ID:
            free(_data);
            break;
        case CPU_DEVICE_MMAP_ID:
            munmap(_data, _size);
            break;
        default:
            CUDA_CALL(cudaFree(_data));
    }
}

std::shared_ptr<Tensor> Tensor::FromMmap(std::string filepath, DataType dtype, 
                                         std::vector<size_t> shape, int device,
                                         std::string name, cudaStream_t stream) {
    auto tensor = std::make_shared<Tensor>();
    size_t expected_size = std::accumulate(shape.begin(), shape.end(), 1ul, std::multiplies<size_t>()) * getDataTypeLength(dtype);
    
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
        SAM_LOG(DEBUG) << "FromMmap: " << name << " cuda malloc " << toReadableSize(size);
        if (stream) {
            CUDA_CALL(cudaMemcpyAsync(d_data, data, size, cudaMemcpyHostToDevice, stream));
            CUDA_CALL(cudaStreamSynchronize(stream));
        } else {
            CUDA_CALL(cudaMemcpy(d_data, data, size, cudaMemcpyHostToDevice));
        }
        munmap(data, size);
        data = d_data;
    } else if (device == CPU_DEVICE_ID){
        SAM_LOG(DEBUG) << "FromMmap: " << name << " cpu malloc " << toReadableSize(size);
        void *new_data = malloc(size);
        memcpy(new_data, data, size);
        munmap(data, size);
        data = new_data;
    } else {
        SAM_LOG(DEBUG) << "FromMmap: mmap " << toReadableSize(size);
    }

    tensor->_dtype = dtype;
    tensor->_size = size;
    tensor->_shape = shape;
    tensor->_data = data;
    tensor->_device = device;
    tensor->_name = name;

    return tensor;
}

std::shared_ptr<Tensor> Tensor::Empty(DataType dtype, std::vector<size_t> shape, int device, std::string name) {
    auto tensor = std::make_shared<Tensor>();
    SAM_CHECK_GT(shape.size(), 0);
    size_t size = std::accumulate(shape.begin(), shape.end(), 1ul, std::multiplies<size_t>()) * getDataTypeLength(dtype);
    
    void *data;
    if (device > CPU_DEVICE_ID) {
        CUDA_CALL(cudaSetDevice(device));
        CUDA_CALL(cudaMalloc(&data, size));
        SAM_LOG(DEBUG) << "Empty: " << name << " cuda " << device << " malloc " << toReadableSize(size) << " with addr " << data;
    } else if (device == CPU_DEVICE_ID) {
        data = malloc(size);
        SAM_LOG(DEBUG) << "Empty: " << name << " cpu malloc " << toReadableSize(size) << " with addr " << data;
    } else {
        SAM_CHECK(0) << "Unvalid device ID";
    }

    tensor->_dtype = dtype;
    tensor->_shape = shape;
    tensor->_size = size;
    tensor->_data = data;
    tensor->_device = device;
    tensor->_name = name;
    
    return tensor;
}

std::shared_ptr<Tensor> Tensor::CreateCopy1D(std::shared_ptr<Tensor> src, size_t item_offset,
                                         std::vector<size_t> shape, std::string name,
                                         cudaStream_t stream) {
    SAM_CHECK(src && src->defined());
    SAM_CHECK_GT(shape.size(), 0);
    auto tensor = std::make_shared<Tensor>();
    size_t size = std::accumulate(shape.begin(), shape.end(), 1ul, std::multiplies<size_t>()) * getDataTypeLength(src ->_dtype);

    tensor->_dtype = src->_dtype;
    tensor->_shape = shape;
    tensor->_size = size;
    
    size_t copy_offset = item_offset * std::accumulate(shape.begin() + 1, shape.end(), 1ul, std::multiplies<size_t>())  * getDataTypeLength(src ->_dtype);
    SAM_CHECK_LE(copy_offset + size, src->_size);

    auto device = src->_device;
    auto src_data = static_cast<const void *>((static_cast<char *>(src->_data) + copy_offset));
    void *data;
    if (device > CPU_DEVICE_ID) {
        CUDA_CALL(cudaSetDevice(device));
        CUDA_CALL(cudaMalloc(&data, size));
        if (stream) {
            CUDA_CALL(cudaMemcpyAsync(data, src_data, size, cudaMemcpyDeviceToDevice, stream));
            CUDA_CALL(cudaStreamSynchronize(stream));
        } else {
            CUDA_CALL(cudaMemcpy(data, src_data, size, cudaMemcpyDeviceToDevice));
        }
        SAM_LOG(DEBUG) << "CreateCopy1D: " << name << " cuda " << device << " malloc " << toReadableSize(size);
    } else {
        data = malloc(tensor->_size);
        memcpy(data, src_data, size);
        device = CPU_DEVICE_ID;
        SAM_LOG(DEBUG) << "CreateCopy1D: " << name << " cpu malloc " << toReadableSize(size);
    }

    tensor->_data = data;
    tensor->_device = device;
    tensor->_name = name;

    return tensor;
}

std::shared_ptr<Tensor> Tensor::FromBlob(void *data, DataType dtype, std::vector<size_t> shape, int device, std::string name) {
    auto tensor = std::make_shared<Tensor>();
    size_t size = std::accumulate(shape.begin(), shape.end(), 1ul, std::multiplies<size_t>()) * getDataTypeLength(dtype);
    
    tensor->_dtype = dtype;
    tensor->_shape = shape;
    tensor->_size = size;
    tensor->_data = data;
    tensor->_device = device;
    tensor->_name = name;

    return tensor;
}

uint64_t encodeBatchKey(uint64_t epoch_idx, uint64_t batch_idx) {
    return ((epoch_idx << Config::kEpochOffset) | (batch_idx << Config::kBatchOffset));
}

uint64_t encodeGraphID(uint64_t key, uint64_t graph_id) {
    return (key & Config::kBatchKeyMask) | (graph_id & Config::kGraphKeyMask);
}

uint64_t decodeBatchKey(uint64_t key) {
    return key & Config::kBatchKeyMask;
}

uint64_t decodeGraphID(uint64_t key) {
    return key & Config::kGraphKeyMask;
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
