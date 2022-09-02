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

#ifndef SAMGRAPH_COMMON_H
#define SAMGRAPH_COMMON_H

#define SAMGRAPH_COLL_CACHE_ENABLE
// #define SAMGRAPH_COLL_CACHE_VALIDATE

#ifdef SAMGRAPH_COLL_CACHE_VALIDATE

#define SAMGRAPH_COLL_CACHE_ENABLE
#define SAMGRAPH_LEGACY_CACHE_ENABLE

#else

#ifndef SAMGRAPH_COLL_CACHE_ENABLE
#define SAMGRAPH_LEGACY_CACHE_ENABLE
#endif

#endif

#include <atomic>
#include <cstdint>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <ostream>
#include <random>
#include <string>
#include <vector>

#include "logging.h"
#include "constant.h"

namespace samgraph {
namespace common {

enum DataType {
  kF32 = 0,
  kF64 = 1,
  kF16 = 2,
  kU8 = 3,
  kI32 = 4,
  kI8 = 5,
  kI64 = 6,
};

enum DeviceType { kCPU = 0, kMMAP = 1, kGPU = 2, kGPU_UM = 3};

enum SampleType {
  kKHop0 = 0,  // vertex-parallel
  kKHop1,      // sample-parallel
  kWeightedKHop,
  kRandomWalk,
  kWeightedKHopPrefix,
  kKHop2,
  kWeightedKHopHashDedup,
  kSaint,
};

// arch0: vanilla mode(CPU sampling + GPU training)
// arch1: standalone mode (single GPU for both sampling and training)
// arch2: offload mode (offload the feature extraction to CPU)
// arch3: dedicated mode (dedicated GPU for sampling and training)
// TODO:  is it a prefetch mode ?
// arch4: prefetch mode
// arch5: distributed mode (CPU/GPU sampling + multi-GPUs traning)
// arch6: sgnn mode
// arch7: sgnn mode but use pytorch extracting
// arch9 : base on arch5 but store single graph on multi-gpu um
enum RunArch {
  kArch0 = 0,
  kArch1,
  kArch2,
  kArch3,
  kArch4,
  kArch5,
  kArch6,
  kArch7,
  kArch9 = 9,
};

// cache by degree: cache the nodes with large degree
// cache by heuristic: cache the training set and the first hop neighbors first,
// then the nodes with large degree
enum CachePolicy {
  kCacheByDegree = 0,
  kCacheByHeuristic,
  kCacheByPreSample,
  kCacheByDegreeHop,
  kCacheByPreSampleStatic,
  kCacheByFakeOptimal,
  kDynamicCache,
  kCacheByRandom,
  kCollCache,
  kCollCacheIntuitive,
  kPartitionCache,
  kPartRepCache,
  kRepCache,
  kCollCacheAsymmLink,
  kCliquePart,
  kCliquePartByDegree,
};

enum class UMPolicy {
  kDegree = 0,
  kTrainset,
  kRandom,
  kPreSample,
  kDefault,
};

enum NegativeSampleType {
  kUniform = 0,
  kSAGELike,
};

enum RollingPolicy {
  AutoRolling = 0,
  EnableRolling,
  DisableRolling,
};

struct Context {
  DeviceType device_type;
  int device_id;

  Context() {}
  Context(DeviceType type, int id) : device_type(type), device_id(id) {}
  Context(std::string name);
  int GetCudaDeviceId() const {
    if (device_type == DeviceType::kCPU || device_type == DeviceType::kMMAP) {
      return -1;
    } else {
      return device_id;
    }
  }
  bool operator==(const Context& rhs) {
    return this->device_type == rhs.device_type &&
           this->device_id == rhs.device_id;
  }
  friend std::ostream& operator<<(std::ostream& os, const Context& ctx) {
    switch (ctx.device_type)
    {
    case DeviceType::kMMAP:
      os << "mmap:" << ctx.device_id;
      return os;
    case DeviceType::kCPU:
      os << "cpu:" << ctx.device_id;    
      return os;
    case DeviceType::kGPU:
      os << "gpu:" << ctx.device_id;
      return os;
    case DeviceType::kGPU_UM:
      os << "gpu_um:" << ctx.device_id;
      return os;
    default:
      LOG(FATAL) << "not support device type "
                 << static_cast<int>(ctx.device_type) << ":" << ctx.device_id;
      // os << "not supprt:" << static_cast<int>(ctx.device_type) << ":" << ctx.device_id;
      return os;
    }
  }
};

using StreamHandle = void*;

class Tensor;
using TensorPtr = std::shared_ptr<Tensor>;

size_t GetDataTypeBytes(DataType dtype);
class Tensor {
 public:
  Tensor();
  ~Tensor();

  inline std::string Name() const { return _name; }
  bool Defined() const { return _data; }
  DataType Type() const { return _dtype; }
  const std::vector<size_t>& Shape() const { return _shape; }
  const void* Data() const { return _data; }
  template<typename T> T* Ptr(){ 
    CHECK(_data == nullptr || (sizeof(T) == GetDataTypeBytes(_dtype))); 
    return static_cast<T*>(_data);
  }
  template<typename T> const T* CPtr() const { return const_cast<Tensor*>(this)->Ptr<T>(); }
  void* MutableData() { return _data; }
  void ReplaceData(void* data);
  void Swap(TensorPtr tensor);
  size_t NumBytes() const { return _nbytes; }
  Context Ctx() const { return _ctx; }
  inline size_t NumItem() const { return std::accumulate(_shape.begin(), _shape.end(), 1ul, std::multiplies<size_t>()); }
  // if the allocated space fits, scale the tensor to `shape` in-place. otherwise allocate a new one.
  void Scale(DataType dt, std::vector<size_t> shape, Context ctx, std::string name);
  // force scale the tensor without checking whether fits, since the size querying may be slow.
  void ForceScale(DataType dt, std::vector<size_t> shape, Context ctx, std::string name);
  void ReShape(std::vector<size_t> new_shape);

  static TensorPtr Null();

  static TensorPtr CreateShm(std::string shm_path, DataType dtype,
                             std::vector<size_t> shape, std::string name);
  static TensorPtr OpenShm(std::string shm_path, DataType dtype,
                             std::vector<size_t> shape, std::string name);
  static TensorPtr Empty(DataType dtype, std::vector<size_t> shape, Context ctx,
                         std::string name);
  static TensorPtr EmptyNoScale(DataType dtype, std::vector<size_t> shape,
                                Context ctx, std::string name);
  static TensorPtr Copy1D(TensorPtr tensor, size_t item_offset,
                          std::vector<size_t> shape, std::string name,
                          StreamHandle stream = nullptr);
  static TensorPtr FromMmap(std::string filepath, DataType dtype,
                            std::vector<size_t> shape, Context ctx,
                            std::string name, StreamHandle stream = nullptr);
  static TensorPtr UMFromMmap(std::string filepath, DataType dtype,
                              std::vector<size_t> shape, std::vector<Context> ctxes,
                              std::string name, std::vector<StreamHandle> streams = {});
  static TensorPtr FromBlob(void* data, DataType dtype,
                            std::vector<size_t> shape, Context ctx,
                            std::string name);
  static TensorPtr CopyTo(TensorPtr source, Context ctx, StreamHandle stream = nullptr, double scale = Constant::kAllocScale);
  static TensorPtr CopyTo(TensorPtr source, Context ctx, StreamHandle stream, std::string name, double scale = Constant::kAllocScale);
  static TensorPtr CopyLine(TensorPtr source, size_t line_idx, Context ctx, StreamHandle stream = nullptr, double scale = Constant::kAllocScale);
  static TensorPtr UMCopyTo(TensorPtr source, std::vector<Context> ctxes, std::vector<StreamHandle> streams = {});
  static TensorPtr UMCopyTo(TensorPtr source, std::vector<Context> ctxes, std::vector<StreamHandle> streams, std::string name);
  static TensorPtr CopyBlob(const void * data, DataType dtype,
                            std::vector<size_t> shape, Context from_ctx,
                            Context to_ctx, std::string name, StreamHandle stream = nullptr);

 private:
  void* _data;
  DataType _dtype;
  Context _ctx;

  size_t _nbytes;
  std::vector<size_t> _shape;

  std::string _name;
};

// Graph dataset that should be loaded from the disk using MMAP.
struct Dataset {
  // Graph topology data
  TensorPtr indptr;
  TensorPtr indices;
  size_t num_node;
  size_t num_edge;

  TensorPtr prob_table;
  TensorPtr alias_table;

  TensorPtr prob_prefix_table;

  TensorPtr in_degrees;
  TensorPtr out_degrees;

  // Decide nodes' feature store in GPU or CPU
  TensorPtr ranking_nodes;
  // for coll cache
  TensorPtr ranking_nodes_list;
  TensorPtr ranking_nodes_freq_list;
  TensorPtr nid_to_block, block_placement;
  TensorPtr block_access_advise;

  // Node feature and label
  size_t num_class;
  TensorPtr feat;
  TensorPtr label;

  // Node set
  TensorPtr train_set;
  TensorPtr test_set;
  TensorPtr valid_set;

  // scale factor for batch feature
  TensorPtr scale_factor;
};

// Train graph in COO format
struct TrainGraph {
  TensorPtr row;
  TensorPtr col;
  TensorPtr data;
  size_t num_src;
  size_t num_dst;
  size_t num_edge;
  TensorPtr src_degree;
  TensorPtr dst_degree;
};

struct MissCacheIndex {
  TensorPtr miss_src_index = nullptr;
  TensorPtr miss_dst_index = nullptr;
  TensorPtr cache_src_index = nullptr;
  TensorPtr cache_dst_index = nullptr;
  size_t num_miss = 0;
  size_t num_cache = 0;
};

struct Task {
  // Key of the task
  uint64_t key;
  // Output graph tensor
  std::vector<std::shared_ptr<TrainGraph>> graphs;
  // node ids of the last train graph
  TensorPtr input_nodes;
  // node ids of the first train graph
  TensorPtr output_nodes;
  // Input feature tensor
  TensorPtr input_feat;
  // Output label tensor
  TensorPtr output_label;
  // combined pos & neg graph for unsupervised learning
  std::shared_ptr<TrainGraph> unsupervised_graph = nullptr;
  size_t unsupervised_positive_edges = 0;
#ifdef SAMGRAPH_LEGACY_CACHE_ENABLE
  // Multi-gpu miss cache index
  MissCacheIndex miss_cache_index;
#endif
  std::atomic_bool graph_remapped;
  Task() : graph_remapped(false) {}
};

using GraphBatch = Task;
using TaskPtr = std::shared_ptr<Task>;

typedef void (*LoopFunction)();
typedef bool (*LoopOnceFunction)();

constexpr static int CPU_CUDA_HOST_MALLOC_DEVICE = 0;
constexpr static int CPU_CLIB_MALLOC_DEVICE = 1;

constexpr static int MMAP_RO_DEVICE = 0;
constexpr static int MMAP_RW_DEVICE = 1;

Context CPU(int device_id = CPU_CUDA_HOST_MALLOC_DEVICE);
Context CPU_CLIB(int device_id = CPU_CLIB_MALLOC_DEVICE);
Context GPU(int device_id = 0);
Context GPU_UM(int device_id = 0);
Context MMAP(int device_id = MMAP_RO_DEVICE);

DataType DataTypeParseName(std::string name);
size_t GetDataTypeBytes(DataType dtype);
size_t GetTensorBytes(DataType dtype, const std::vector<size_t> shape);
size_t GetTensorBytes(DataType dtype,
                      std::vector<size_t>::const_iterator shape_start,
                      std::vector<size_t>::const_iterator shape_end);
// predict the number of sampled nodes
size_t PredictNumNodes(size_t batch_size, const std::vector<size_t>& fanout,
                       size_t num_fanout_to_comp);
size_t PredictNumRandomWalkEdges(size_t batch_size,
                                 const std::vector<size_t>& fanout,
                                 size_t num_fanout_to_comp,
                                 size_t num_random_walk,
                                 size_t random_walk_length);

std::string ToReadableSize(size_t nbytes);
std::string ToPercentage(double percentage);

std::string GetEnv(std::string key);
bool IsEnvSet(std::string key);
std::string GetTimeString();
bool FileExist(const std::string& filepath);

template <typename T>
inline T RoundUpDiv(T target, T unit) {
  return (target + unit - 1) / unit;
}

template <typename T>
inline T RoundUp(T target, T unit) {
  return RoundUpDiv(target, unit) * unit;
}

template <typename T>
inline T Max(T a, T b) {
  return a > b ? a : b;
}

template <typename T>
inline T Min(T a, T b) {
  return a < b ? a : b;
}

template < typename T >
T GCD(T a, T b) {
  if(b) while((a %= b) && (b %= a));
  return a + b;
}
template < typename T >
T LCM(T a, T b) {
  return a * b / GCD(a, b);
}

// shuffler virtual class
class Shuffler {
 public:
  virtual ~Shuffler() {};
  virtual TensorPtr GetBatch(StreamHandle stream = nullptr) = 0;
  virtual uint64_t Epoch() = 0;
  virtual uint64_t Step() = 0;
  virtual uint64_t LocalStep() { CHECK(false) << "Unimplemented"; return 0; };
  virtual size_t NumEpoch() = 0;
  virtual size_t NumStep() = 0;
  virtual size_t NumLocalStep() { CHECK(false) << "Unimplemented"; return 0; };
  virtual void Reset() {};
};

std::ostream& operator<<(std::ostream&, const SampleType);
std::ostream& operator<<(std::ostream&, const CachePolicy);

}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_COMMON_H
