#include "common.h"

#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <chrono>  // chrono::system_clock
#include <cstdlib>
#include <cstring>
#include <ctime>    // localtime
#include <iomanip>  // put_time
#include <numeric>
#include <sstream>  // stringstream
#include <string>   // string

#include "constant.h"
#include "device.h"
#include "logging.h"

namespace samgraph {
namespace common {

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
  CHECK(FileExist(filepath));

  TensorPtr tensor = std::make_shared<Tensor>();
  size_t nbytes = GetTensorBytes(dtype, shape.begin(), shape.end());

  struct stat st;
  stat(filepath.c_str(), &st);
  size_t file_nbytes = st.st_size;
  CHECK_EQ(nbytes, file_nbytes);

  int fd = open(filepath.c_str(), O_RDONLY, 0);
  void *data = mmap(NULL, nbytes, PROT_READ, MAP_SHARED | MAP_FILE, fd, 0);
  CHECK_NE(data, (void *)-1);
  mlock(data, nbytes);
  close(fd);

  tensor->_dtype = dtype;
  tensor->_nbytes = nbytes;
  tensor->_shape = shape;
  tensor->_ctx = ctx;
  tensor->_name = name;

  // if the device is cuda, we have to copy the data from host memory to cuda
  // memory
  switch (ctx.device_type) {
    case kCPU:
    case kGPU:
      tensor->_data = Device::Get(ctx)->AllocWorkspace(ctx, nbytes,
                                                       Constant::kAllocNoScale);
      Device::Get(ctx)->CopyDataFromTo(data, 0, tensor->_data, 0, nbytes, CPU(),
                                       ctx, stream);
      Device::Get(ctx)->StreamSync(ctx, stream);
      munmap(data, nbytes);
      break;
    case kMMAP:
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

  if (ctx.device_type == kMMAP) {
    ctx = CPU();
  }

  tensor->_dtype = dtype;
  tensor->_shape = shape;
  tensor->_nbytes = nbytes;
  tensor->_data = Device::Get(ctx)->AllocWorkspace(ctx, nbytes);
  tensor->_ctx = ctx;
  tensor->_name = name;

  return tensor;
}

TensorPtr Tensor::Copy1D(TensorPtr source, size_t item_offset,
                         std::vector<size_t> shape, std::string name,
                         StreamHandle stream) {
  CHECK(source && source->Defined());
  CHECK_GT(shape.size(), 0);

  TensorPtr output = std::make_shared<Tensor>();
  size_t nbytes = GetTensorBytes(source->_dtype, shape.begin(), shape.end());

  Context output_ctx = source->_ctx;
  if (output_ctx.device_type == kMMAP) {
    output_ctx = CPU();
  }

  size_t copy_start_offset =
      item_offset *
      GetTensorBytes(source->_dtype, shape.begin() + 1, shape.end());

  CHECK_LE(copy_start_offset + nbytes, source->_nbytes);

  output->_dtype = source->_dtype;
  output->_shape = shape;
  output->_nbytes = nbytes;
  output->_ctx = output_ctx;
  output->_data =
      Device::Get(output->_ctx)->AllocWorkspace(output->_ctx, nbytes);
  output->_name = name;

  Device::Get(output->_ctx)
      ->CopyDataFromTo(source->_data, copy_start_offset, output->_data, 0,
                       nbytes, source->_ctx, output->_ctx, stream);

  Device::Get(output->_ctx)->StreamSync(output->_ctx, stream);

  return output;
}

Context CPU(int device_id) { return {kCPU, device_id}; }
Context GPU(int device_id) { return {kGPU, device_id}; }
Context MMAP(int device_id) { return {kMMAP, device_id}; }

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

std::string ToReadableSize(size_t nbytes) {
  char buf[Constant::kBufferSize];
  if (nbytes > Constant::kGigabytes) {
    double new_size = (float)nbytes / Constant::kGigabytes;
    sprintf(buf, "%.2lf GB", new_size);
    return std::string(buf);
  } else if (nbytes > Constant::kMegabytes) {
    double new_size = (float)nbytes / Constant::kMegabytes;
    sprintf(buf, "%.2lf MB", new_size);
    return std::string(buf);
  } else if (nbytes > Constant::kKilobytes) {
    double new_size = (float)nbytes / Constant::kKilobytes;
    sprintf(buf, "%.2lf KB", new_size);
    return std::string(buf);
  } else {
    double new_size = (float)nbytes;
    sprintf(buf, "%.2lf Bytes", new_size);
    return std::string(buf);
  }
}

std::string ToPercentage(double percentage) {
  char buf[Constant::kBufferSize];
  sprintf(buf, "%.2lf %%", percentage * 100);
  return std::string(buf);
}

size_t GetDataTypeBytes(DataType dtype) {
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

size_t GetTensorBytes(DataType dtype, const std::vector<size_t> shape) {
  return std::accumulate(shape.begin(), shape.end(), 1ul,
                         std::multiplies<size_t>()) *
         GetDataTypeBytes(dtype);
}

size_t GetTensorBytes(DataType dtype,
                      std::vector<size_t>::const_iterator shape_start,
                      std::vector<size_t>::const_iterator shape_end) {
  return std::accumulate(shape_start, shape_end, 1ul,
                         std::multiplies<size_t>()) *
         GetDataTypeBytes(dtype);
}

size_t PredictNumNodes(size_t batch_size, const std::vector<int> &fanout,
                       size_t num_fanout_to_comp) {
  CHECK_LE(num_fanout_to_comp, fanout.size());
  size_t predicted_num_nodes = 0;
  size_t cur_layer_num_nodes = batch_size;
  for (size_t i = 0; i < num_fanout_to_comp; i++) {
    predicted_num_nodes += cur_layer_num_nodes;
    cur_layer_num_nodes *= fanout[i];
  }

  predicted_num_nodes += cur_layer_num_nodes;
  return predicted_num_nodes;
}

std::string GetEnv(std::string key) {
  const char *env_var_val = getenv(key.c_str());
  if (env_var_val != nullptr) {
    return std::string(env_var_val);
  } else {
    return "";
  }
}

bool IsEnvSet(std::string key) {
  std::string val = GetEnv(key);
  if (val == "ON" || val == "1") {
    LOG(INFO) << "Environment variable " << key << " is set to " << val;
    return true;
  } else {
    return false;
  }
}

std::string GetTimeString() {
  auto now = std::chrono::system_clock::now();
  auto in_time_t = std::chrono::system_clock::to_time_t(now);

  std::stringstream ss;
  ss << std::put_time(std::localtime(&in_time_t), "%Y%m%dT%H%M%S%");
  return ss.str();
}

bool FileExist(const std::string &filepath) {
  std::ifstream f(filepath);
  return f.good();
}

}  // namespace common
}  // namespace samgraph
