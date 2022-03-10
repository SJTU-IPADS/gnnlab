/*
 * Copyright 2022 Institute of Parallel and Distributed Systems, Shanghai Jiao Tong University
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include "common.h"

#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cctype>
#include <chrono>  // chrono::system_clock
#include <cstddef>
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

Context::Context(std::string name) {
  size_t delim_pos = name.find(':');
  CHECK_NE(delim_pos, std::string::npos);

  std::string device_str = name.substr(0, delim_pos);
  std::string id_str = name.substr(delim_pos + 1, std::string::npos);
  CHECK(device_str == "cpu" || device_str == "cuda" || device_str == "mmap");

  if (device_str == "cpu") {
    device_type = kCPU;
  } else if (device_str == "cuda") {
    device_type = kGPU;
  } else if (device_str == "mmap") {
    device_type = kMMAP;
  } else {
    CHECK(false);
  }

  device_id = std::stoi(id_str);
}

Tensor::Tensor() : _data(nullptr) {}

Tensor::~Tensor() {
  if (!_data) {
    return;
  }

  Device::Get(_ctx)->FreeWorkspace(_ctx, _data, _nbytes);
  LOG(DEBUG) << "Tensor " << _name << " has been freed";
}

void Tensor::ReplaceData(void *data) {
  Device::Get(_ctx)->FreeWorkspace(_ctx, _data);
  _data = data;
}

void Tensor::Swap(TensorPtr tensor) {
  CHECK(this->Ctx() == tensor->Ctx());
  CHECK(this->Shape() == tensor->Shape());
  CHECK(this->Type() == tensor->Type());
  std::swap(this->_data, tensor->_data);
}

void Tensor::ChangeShape(DataType dt, std::vector<size_t> shape, Context ctx, std::string name) {
  if (!Defined()) {
    CHECK_GT(shape.size(), 0);
    size_t nbytes = GetTensorBytes(dt, shape.begin(), shape.end());

    if (ctx.device_type == kMMAP) {
      ctx = CPU(ctx.device_id);
    }

    _dtype = dt;
    _shape = shape;
    _nbytes = nbytes;
    _data = Device::Get(ctx)->AllocWorkspace(ctx, nbytes);
    _ctx = ctx;
    _name = name;
    return;
  } 
  CHECK(_dtype == dt);
  CHECK(_ctx == ctx);
  CHECK(_shape.size() == shape.size());
  if (Device::Get(ctx)->WorkspaceActualSize(ctx, _data) < GetTensorBytes(dt, shape)) {
    Device::Get(ctx)->FreeWorkspace(ctx, _data);
    _data = Device::Get(ctx)->AllocWorkspace(ctx, GetTensorBytes(dt, shape));
  }
  _name = name;
  _shape = shape;
  _nbytes = GetTensorBytes(dt, shape.begin(), shape.end());
}
void Tensor::ForceChangeShape(DataType dt, std::vector<size_t> shape, Context ctx, std::string name) {
  CHECK(Defined());
  CHECK(_dtype == dt);
  CHECK(_ctx == ctx);
  CHECK(_shape.size() == shape.size());
  _name = name;
  _shape = shape;
  _nbytes = GetTensorBytes(dt, shape.begin(), shape.end());
}

TensorPtr Tensor::Null() { return std::make_shared<Tensor>(); }

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
  void *data = mmap(NULL, nbytes, PROT_READ, MAP_SHARED | MAP_FILE | MAP_LOCKED, fd, 0);
  CHECK_NE(data, (void *)-1);
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
    ctx = CPU(ctx.device_id);
  }

  tensor->_dtype = dtype;
  tensor->_shape = shape;
  tensor->_nbytes = nbytes;
  tensor->_data = Device::Get(ctx)->AllocWorkspace(ctx, nbytes);
  tensor->_ctx = ctx;
  tensor->_name = name;

  return tensor;
}
TensorPtr Tensor::EmptyNoScale(DataType dtype, std::vector<size_t> shape,
                               Context ctx, std::string name) {
  TensorPtr tensor = std::make_shared<Tensor>();
  CHECK_GT(shape.size(), 0);
  size_t nbytes = GetTensorBytes(dtype, shape.begin(), shape.end());

  if (ctx.device_type == kMMAP) {
    ctx = CPU(ctx.device_id);
  }

  tensor->_dtype = dtype;
  tensor->_shape = shape;
  tensor->_nbytes = nbytes;
  tensor->_data = Device::Get(ctx)->
    AllocWorkspace(ctx, nbytes, Constant::kAllocNoScale);
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
TensorPtr Tensor::CopyBlob(const void *data, DataType dtype,
                           std::vector<size_t> shape,
                           Context from_ctx, Context to_ctx,
                           std::string name, StreamHandle stream) {
  TensorPtr tensor = std::make_shared<Tensor>();
  size_t nbytes = GetTensorBytes(dtype, shape.begin(), shape.end());
  tensor->_data =
      Device::Get(to_ctx)->AllocWorkspace(to_ctx, nbytes);
  if (to_ctx.device_type == kGPU) {
    Device::Get(to_ctx)
      ->CopyDataFromTo(data, 0, tensor->_data, 0,
                       nbytes, from_ctx, to_ctx, stream);
  } else {
    Device::Get(from_ctx)
      ->CopyDataFromTo(data, 0, tensor->_data, 0,
                       nbytes, from_ctx, to_ctx, stream);
  }
  tensor->_dtype = dtype;
  tensor->_shape = shape;
  tensor->_nbytes = nbytes;
  tensor->_ctx = to_ctx;
  tensor->_name = name;

  return tensor;
}

TensorPtr Tensor::CopyTo(TensorPtr source, Context ctx, StreamHandle stream) {
  CHECK(source && source->Defined());
  std::vector<size_t> shape = source->Shape();
  auto dtype = source->_dtype;
  CHECK_GT(shape.size(), 0);

  TensorPtr tensor = std::make_shared<Tensor>();
  size_t nbytes = GetTensorBytes(dtype, shape.begin(), shape.end());

  tensor->_dtype = source->_dtype;
  tensor->_shape = shape;
  tensor->_nbytes = source->_nbytes;
  tensor->_ctx = ctx;
  tensor->_data =
      Device::Get(ctx)->AllocWorkspace(ctx, nbytes, Constant::kAllocNoScale);
  tensor->_name = source->_name;
  if (source->Ctx().device_type == kGPU && ctx.device_type != kGPU) {
    Device::Get(source->Ctx())->CopyDataFromTo(source->_data, 0, tensor->_data, 0, nbytes, source->_ctx, tensor->_ctx, stream);
  } else {
    Device::Get(ctx)->CopyDataFromTo(source->_data, 0, tensor->_data, 0,
                                    nbytes, source->_ctx, tensor->_ctx, stream);
  }
  Device::Get(tensor->_ctx)->StreamSync(tensor->_ctx, stream);

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

size_t PredictNumNodes(size_t batch_size, const std::vector<size_t> &fanout,
                       size_t num_fanout_to_comp) {
  CHECK_LE(num_fanout_to_comp, fanout.size());
  size_t count = batch_size;
  for (int i = num_fanout_to_comp - 1; i >= 0; i--) {
    count += (count * fanout[i]);
  }

  return count;
}

size_t PredictNumRandomWalkEdges(size_t batch_size,
                                 const std::vector<size_t> &fanout,
                                 size_t num_fanout_to_comp,
                                 size_t num_random_walk,
                                 size_t random_walk_length) {
  size_t num_nodes = PredictNumNodes(batch_size, fanout, num_fanout_to_comp);
  return num_nodes * num_random_walk * random_walk_length;
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

std::ostream &operator<<(std::ostream &os, const SampleType type) {
  switch (type) {
    case kKHop0:
      os << "KHop0";
      break;
    case kKHop1:
      os << "KHop1";
      break;
    case kWeightedKHop:
      os << "WeightedKHop";
      break;
    case kRandomWalk:
      os << "RandomWalk";
      break;
    case kWeightedKHopPrefix:
      os << "WeightedKHopPrefix";
      break;
    case kKHop2:
      os << "KHop2";
      break;
    case kWeightedKHopHashDedup:
      os << "WeightedKHopHashDedup";
      break;
    default:
      CHECK(false);
  }

  return os;
}

std::ostream& operator<<(std::ostream& os, const CachePolicy policy) {
  std::string str;
  switch (policy) {
    case kCacheByDegree:
      os << "degree";
      break;
    case kCacheByHeuristic:
      os << "heuristic";
      break;
    case kCacheByPreSample:
      os << "preSample";
      break;
    case kCacheByPreSampleStatic:
      os << "preSampleStatic";
      break;
    case kCacheByDegreeHop:
      os << "degree_hop";
      break;
    case kCacheByFakeOptimal:
      os << "fake_optimal";
      break;
    case kCacheByRandom:
      os << "random";
      break;
    default:
      CHECK(false);
  }

  return os;
}

}  // namespace common
}  // namespace samgraph
