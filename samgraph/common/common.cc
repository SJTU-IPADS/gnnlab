#include "common.h"

#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <numeric>

#include "config.h"
#include "logging.h"

namespace samgraph {
namespace common {

Tensor::Tensor() : _data(nullptr), _device(CPU_DEVICE_ID) {}

Tensor::~Tensor() {
  if (!_data) {
    return;
  }

  LOG(DEBUG) << "Tensor " << _name << " has been freed";
  switch (_device) {
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

TensorPtr Tensor::FromMmap(std::string filepath, DataType dtype,
                           std::vector<size_t> shape, int device,
                           std::string name, cudaStream_t stream) {
  auto tensor = std::make_shared<Tensor>();
  size_t expected_size = std::accumulate(shape.begin(), shape.end(), 1ul,
                                         std::multiplies<size_t>()) *
                         GetDataTypeLength(dtype);

  struct stat st;
  stat(filepath.c_str(), &st);
  size_t size = st.st_size;
  CHECK_EQ(size, expected_size);

  int fd = open(filepath.c_str(), O_RDONLY, 0);
  void *data = mmap(NULL, size, PROT_READ, MAP_SHARED | MAP_FILE, fd, 0);
  mlock(data, size);
  close(fd);

  // if the device is cuda, we have to copy the data from host memory to cuda
  // memory
  if (device > CPU_DEVICE_ID) {
    void *d_data;
    CUDA_CALL(cudaSetDevice(device));
    CUDA_CALL(cudaMalloc(&d_data, size));
    LOG(DEBUG) << "FromMmap: " << name << " cuda malloc "
               << ToReadableSize(size);
    if (stream) {
      CUDA_CALL(
          cudaMemcpyAsync(d_data, data, size, cudaMemcpyHostToDevice, stream));
      CUDA_CALL(cudaStreamSynchronize(stream));
    } else {
      CUDA_CALL(cudaMemcpy(d_data, data, size, cudaMemcpyHostToDevice));
    }
    munmap(data, size);
    data = d_data;
  } else if (device == CPU_DEVICE_ID) {
    LOG(DEBUG) << "FromMmap: " << name << " cpu malloc "
               << ToReadableSize(size);
    void *new_data = malloc(size);
    memcpy(new_data, data, size);
    munmap(data, size);
    data = new_data;
  } else {
    LOG(DEBUG) << "FromMmap: mmap " << ToReadableSize(size);
  }

  tensor->_dtype = dtype;
  tensor->_size = size;
  tensor->_shape = shape;
  tensor->_data = data;
  tensor->_device = device;
  tensor->_name = name;

  return tensor;
}

TensorPtr Tensor::Empty(DataType dtype, std::vector<size_t> shape, int device,
                        std::string name) {
  auto tensor = std::make_shared<Tensor>();
  CHECK_GT(shape.size(), 0);
  size_t size = std::accumulate(shape.begin(), shape.end(), 1ul,
                                std::multiplies<size_t>()) *
                GetDataTypeLength(dtype);

  void *data;
  if (device > CPU_DEVICE_ID) {
    CUDA_CALL(cudaSetDevice(device));
    CUDA_CALL(cudaMalloc(&data, size));
    LOG(DEBUG) << "Empty: " << name << " cuda " << device << " malloc "
               << ToReadableSize(size) << " with addr " << data;
  } else if (device == CPU_DEVICE_ID) {
    data = malloc(size);
    LOG(DEBUG) << "Empty: " << name << " cpu malloc " << ToReadableSize(size)
               << " with addr " << data;
  } else {
    CHECK(0) << "Unvalid device ID";
  }

  tensor->_dtype = dtype;
  tensor->_shape = shape;
  tensor->_size = size;
  tensor->_data = data;
  tensor->_device = device;
  tensor->_name = name;

  return tensor;
}

TensorPtr Tensor::CreateCopy1D(TensorPtr src, size_t item_offset,
                               std::vector<size_t> shape, std::string name,
                               cudaStream_t stream) {
  CHECK(src && src->defined());
  CHECK_GT(shape.size(), 0);
  auto tensor = std::make_shared<Tensor>();
  size_t size = std::accumulate(shape.begin(), shape.end(), 1ul,
                                std::multiplies<size_t>()) *
                GetDataTypeLength(src->_dtype);

  tensor->_dtype = src->_dtype;
  tensor->_shape = shape;
  tensor->_size = size;

  size_t copy_offset = item_offset *
                       std::accumulate(shape.begin() + 1, shape.end(), 1ul,
                                       std::multiplies<size_t>()) *
                       GetDataTypeLength(src->_dtype);
  CHECK_LE(copy_offset + size, src->_size);

  auto device = src->_device;
  auto src_data = static_cast<const void *>(
      (static_cast<char *>(src->_data) + copy_offset));
  void *data;
  if (device > CPU_DEVICE_ID) {
    CUDA_CALL(cudaSetDevice(device));
    CUDA_CALL(cudaMalloc(&data, size));
    if (stream) {
      CUDA_CALL(cudaMemcpyAsync(data, src_data, size, cudaMemcpyDeviceToDevice,
                                stream));
      CUDA_CALL(cudaStreamSynchronize(stream));
    } else {
      CUDA_CALL(cudaMemcpy(data, src_data, size, cudaMemcpyDeviceToDevice));
    }
    LOG(DEBUG) << "CreateCopy1D: " << name << " cuda " << device << " malloc "
               << ToReadableSize(size);
  } else {
    data = malloc(tensor->_size);
    memcpy(data, src_data, size);
    device = CPU_DEVICE_ID;
    LOG(DEBUG) << "CreateCopy1D: " << name << " cpu malloc "
               << ToReadableSize(size);
  }

  tensor->_data = data;
  tensor->_device = device;
  tensor->_name = name;

  return tensor;
}

TensorPtr Tensor::FromBlob(void *data, DataType dtype,
                           std::vector<size_t> shape, int device,
                           std::string name) {
  auto tensor = std::make_shared<Tensor>();
  size_t size = std::accumulate(shape.begin(), shape.end(), 1ul,
                                std::multiplies<size_t>()) *
                GetDataTypeLength(dtype);

  tensor->_dtype = dtype;
  tensor->_shape = shape;
  tensor->_size = size;
  tensor->_data = data;
  tensor->_device = device;
  tensor->_name = name;

  return tensor;
}

TensorPtr Tensor::ToDevice(const TensorPtr origin, int device,
                           cudaStream_t stream) {
  auto tensor = std::make_shared<Tensor>();

  tensor->_dtype = origin->_dtype;
  tensor->_shape = origin->_shape;
  tensor->_size = origin->_size;
  tensor->_device = origin->_device;
  tensor->_name = origin->_name + "_ToDevice";

  if (device == CPU_DEVICE_ID) {
    tensor->_data = malloc(origin->_size);
  } else if (device) {
    CUDA_CALL(cudaSetDevice(device));
    CUDA_CALL(cudaMalloc(&tensor->_data, tensor->_size));
  } else {
    CHECK(0);
  }

  if (device == CPU_DEVICE_ID && origin->_device <= CPU_DEVICE_ID) {
    memcpy(tensor->_data, origin->_data, tensor->_size);
  } else {
    cudaMemcpyKind kind;
    if (device == CPU_DEVICE_ID && origin->_device > CPU_DEVICE_ID) {
      kind = cudaMemcpyDeviceToHost;
    } else if (device > CPU_DEVICE_ID && origin->_device <= CPU_DEVICE_ID) {
      kind = cudaMemcpyHostToDevice;
    } else if (device > CPU_DEVICE_ID && origin->_device > CPU_DEVICE_ID) {
      kind = cudaMemcpyDeviceToDevice;
    } else {
      CHECK(0);
    }

    if (stream) {
      CUDA_CALL(cudaMemcpyAsync(tensor->_data, origin->_data, tensor->_size,
                                kind, stream));
      CUDA_CALL(cudaStreamSynchronize(stream));
    } else {
      CUDA_CALL(cudaMemcpy(tensor->_data, origin->_data, tensor->_size, kind));
    }
  }

  return tensor;
}

size_t GetDataTypeLength(int dtype) {
  switch (dtype) {
    case kI8:
    case kU8:
      return 1ul;
    case kF16:
      return 2ul;
    case kF32:
    case kI32:
      return 4ul;
    case kI64:
    case kF64:
      return 8ul;
    default:
      CHECK(0) << "Unsupported data type: " << dtype;
  }
  return 4ul;
}

std::string ToReadableSize(size_t size_in_bytes) {
  char buf[Config::kBufferSize];
  if (size_in_bytes > Config::kGigabytes) {
    double new_size = (float)size_in_bytes / Config::kGigabytes;
    sprintf(buf, "%.2lf GB", new_size);
    return std::string(buf);
  } else if (size_in_bytes > Config::kMegabytes) {
    double new_size = (float)size_in_bytes / Config::kMegabytes;
    sprintf(buf, "%.2lf MB", new_size);
    return std::string(buf);
  } else if (size_in_bytes > Config::kKilobytes) {
    double new_size = (float)size_in_bytes / Config::kKilobytes;
    sprintf(buf, "%.2lf KB", new_size);
    return std::string(buf);
  } else {
    double new_size = (float)size_in_bytes;
    sprintf(buf, "%.2lf Bytes", new_size);
    return std::string(buf);
  }
}

}  // namespace common
}  // namespace samgraph
