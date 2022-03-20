#ifndef SAMGRAPH_COMMON_H
#define SAMGRAPH_COMMON_H

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

namespace samgraph {
namespace common {

using IdType = unsigned int;
using Id64Type = unsigned long long int;

enum DataType {
  kF32 = 0,
  kF64 = 1,
  kF16 = 2,
  kU8 = 3,
  kI32 = 4,
  kI8 = 5,
  kI64 = 6,
};

enum DeviceType { kCPU = 0, kMMAP = 1, kGPU = 2, kGPU_UM= 3 };

enum SampleType {
  kKHop0 = 0,  // vertex-parallel
  kKHop1,      // sample-parallel
  kWeightedKHop,
  kRandomWalk,
  kWeightedKHopPrefix,
  kKHop2,
  kWeightedKHopHashDedup,
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
enum RunArch {
  kArch0 = 0,
  kArch1,
  kArch2,
  kArch3,
  kArch4,
  kArch5,
  kArch6,
  kArch7
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
};

enum class UMPolicy {
  kDegree = 0,
  kTrainset,
  kRandom,
  kPreSample,
  kDefault,
};

struct Context {
  DeviceType device_type;
  int device_id;

  Context() {}
  Context(DeviceType type, int id) : device_type(type), device_id(id) {}
  Context(std::string name);
  bool operator==(const Context& rhs) {
    return this->device_type == rhs.device_type &&
           this->device_id == rhs.device_id;
  }
};

using StreamHandle = void*;

class Tensor;
using TensorPtr = std::shared_ptr<Tensor>;

class Tensor {
 public:
  Tensor();
  ~Tensor();

  bool Defined() const { return _data; }
  DataType Type() const { return _dtype; }
  const std::vector<size_t>& Shape() const { return _shape; }
  const void* Data() const { return _data; }
  void* MutableData() { return _data; }
  void ReplaceData(void* data);
  size_t NumBytes() const { return _nbytes; }
  Context Ctx() const { return _ctx; }
  std::string Name() const { return _name; }

  static TensorPtr Null();
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
  static TensorPtr FromBlob(void* data, DataType dtype,
                            std::vector<size_t> shape, Context ctx,
                            std::string name);
  static TensorPtr CopyTo(TensorPtr source, Context ctx, StreamHandle stream = nullptr);
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

  // Node feature and label
  size_t num_class;
  TensorPtr feat;
  TensorPtr label;

  // Node set
  TensorPtr train_set;
  TensorPtr test_set;
  TensorPtr valid_set;
};

// Train graph in COO format
struct TrainGraph {
  TensorPtr row;
  TensorPtr col;
  TensorPtr data;
  size_t num_src;
  size_t num_dst;
  size_t num_edge;
};

struct MissCacheIndex {
  TensorPtr miss_src_index;
  TensorPtr miss_dst_index;
  TensorPtr cache_src_index;
  TensorPtr cache_dst_index;
  size_t num_miss;
  size_t num_cache;
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
  // Multi-gpu miss cache index
  MissCacheIndex miss_cache_index;
  std::atomic_bool graph_remapped;
  Task() : graph_remapped(false) {}
};

using GraphBatch = Task;
using TaskPtr = std::shared_ptr<Task>;

typedef void (*LoopFunction)();
typedef bool (*LoopOnceFunction)();

constexpr static int CPU_CUDA_HOST_MALLOC_DEVICE = 0;
constexpr static int CPU_CLIB_MALLOC_DEVICE = 1;

Context CPU(int device_id = CPU_CUDA_HOST_MALLOC_DEVICE);
Context CPU_CLIB(int device_id = CPU_CLIB_MALLOC_DEVICE);
Context GPU(int device_id = 0);
Context MMAP(int device_id = 0);

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

// shuffler virtual class
class Shuffler {
 public:
  virtual ~Shuffler() {};
  virtual TensorPtr GetBatch(StreamHandle stream = nullptr) = 0;
  virtual uint64_t Epoch() = 0;
  virtual uint64_t Step() = 0;
  virtual size_t NumEpoch() = 0;
  virtual size_t NumStep() = 0;
  virtual size_t NumLocalStep() { return 0; };
  virtual void Reset() {};
};

std::ostream& operator<<(std::ostream&, const SampleType);
std::ostream& operator<<(std::ostream&, const CachePolicy);

}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_COMMON_H
