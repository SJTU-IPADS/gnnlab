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
#include "device.h"
#include "logging.h"

namespace samgraph {
namespace common {

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

size_t GetTensorBytes(int dtype,
                      std::vector<size_t>::const_iterator shape_start,
                      std::vector<size_t>::const_iterator shape_end) {
  return std::accumulate(shape_start, shape_end, 1ul,
                         std::multiplies<size_t>()) *
         GetDataTypeLength(dtype);
}

Tensor::Tensor() : _data(nullptr) {}

Tensor::~Tensor() {
  if (!_data) {
    return;
  }

  Device::Get(_ctx)->FreeWorkspace(_ctx, _data, _nbytes);
  LOG(DEBUG) << "Tensor " << _name << " has been freed";
}

TensorPtr Tensor::FromMmap(std::string filepath, DataType dtype,
                           std::vector<size_t> shape, Context ctx,
                           std::string name, StreamHandle stream) {
  TensorPtr tensor = std::make_shared<Tensor>();
  size_t nbytes = GetTensorBytes(dtype, shape.begin(), shape.end());

  struct stat st;
  stat(filepath.c_str(), &st);
  size_t file_nbytes = st.st_size;
  CHECK_EQ(nbytes, file_nbytes);

  int fd = open(filepath.c_str(), O_RDONLY, 0);
  void *data = mmap(NULL, nbytes, PROT_READ, MAP_SHARED | MAP_FILE, fd, 0);
  mlock(data, nbytes);
  close(fd);

  tensor->_dtype = dtype;
  tensor->_nbytes = nbytes;
  tensor->_shape = shape;
  tensor->_ctx = ctx;
  tensor->_name = name;

  // if the device is cuda, we have to copy the data from host memory to cuda
  // memory
  if (ctx.device_type != kMMAP) {
    tensor->_data = data;
  } else {
  }
  switch (ctx.device_type) {
    case kGPU:
      tensor->_data = Device::Get(ctx)->AllocWorkspace(ctx, nbytes);
      Device::Get(ctx)->CopyDataFromTo(data, 0, tensor->_data, 0, nbytes, CPU(),
                                       ctx, stream);
      Device::Get(ctx)->StreamSync(ctx, stream);
      munmap(data, nbytes);
      break;
    case kMMAP:
    case kCPU:
      ctx.device_type = kMMAP;
      tensor->_data = data;
      break;
    default:
      CHECK(0);
  }

  return tensor;
}

TensorPtr Tensor::Empty(DataType dtype, std::vector<size_t> shape, Context ctx,
                        std::string name) {
  TensorPtr tensor = std::make_shared<Tensor>();
  CHECK_GT(shape.size(), 0);
  size_t nbytes = GetTensorBytes(dtype, shape.begin(), shape.end());

  tensor->_dtype = dtype;
  tensor->_shape = shape;
  tensor->_nbytes = nbytes;
  tensor->_data = Device::Get(ctx)->AllocWorkspace(ctx, nbytes);
  tensor->_ctx = ctx;
  tensor->_name = name;

  return tensor;
}

TensorPtr Tensor::CreateCopy1D(TensorPtr source, size_t item_offset,
                               std::vector<size_t> shape, std::string name,
                               StreamHandle stream) {
  CHECK(source && source->Defined());
  CHECK_GT(shape.size(), 0);

  TensorPtr output = std::make_shared<Tensor>();
  size_t nbytes = GetTensorBytes(source->_dtype, shape.begin(), shape.end());

  size_t copy_start_offset =
      item_offset *
      GetTensorBytes(source->_dtype, shape.begin() + 1, shape.end());

  CHECK_LE(copy_start_offset + nbytes, source->_nbytes);

  output->_dtype = source->_dtype;
  output->_shape = shape;
  output->_nbytes = nbytes;
  output->_ctx = source->_ctx;
  output->_data =
      Device::Get(output->_ctx)->AllocWorkspace(output->_ctx, nbytes);
  output->_name = name;

  Device::Get(output->_ctx)
      ->CopyDataFromTo(source->_data, copy_start_offset, output->_data, 0,
                       nbytes, source->_ctx, output->_ctx, stream);

  Device::Get(output->_ctx)->StreamSync(output->_ctx, stream);

  return output;
}

TensorPtr Tensor::FromBlob(void *data, DataType dtype,
                           std::vector<size_t> shape, Context ctx,
                           std::string name) {
  TensorPtr tensor = std::make_shared<Tensor>();
  size_t nbytes = GetTensorBytes(dtype, shape.begin(), shape.end());

  tensor->_dtype = dtype;
  tensor->_shape = shape;
  tensor->_nbytes = nbytes;
  tensor->_data = data;
  tensor->_ctx = ctx;
  tensor->_name = name;

  return tensor;
}

Context CPU(int device_id) { return {kCPU, device_id}; }
Context GPU(int device_id) { return {kGPU, device_id}; }
Context MMAP(int device_id) { return {kMMAP, device_id}; }

std::string ToReadableSize(size_t nbytes) {
  char buf[Config::kBufferSize];
  if (nbytes > Config::kGigabytes) {
    double new_size = (float)nbytes / Config::kGigabytes;
    sprintf(buf, "%.2lf GB", new_size);
    return std::string(buf);
  } else if (nbytes > Config::kMegabytes) {
    double new_size = (float)nbytes / Config::kMegabytes;
    sprintf(buf, "%.2lf MB", new_size);
    return std::string(buf);
  } else if (nbytes > Config::kKilobytes) {
    double new_size = (float)nbytes / Config::kKilobytes;
    sprintf(buf, "%.2lf KB", new_size);
    return std::string(buf);
  } else {
    double new_size = (float)nbytes;
    sprintf(buf, "%.2lf Bytes", new_size);
    return std::string(buf);
  }
}

}  // namespace common
}  // namespace samgraph
