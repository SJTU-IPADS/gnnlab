#ifndef SAMGRAPH_COMMON_H
#define SAMGRAPH_COMMON_H

#include <cuda_runtime.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace samgraph {
namespace common {

using IdType = unsigned int;
using SignedIdType = int;

// Keep the order consistent with DMLC/mshadow
// https://github.com/dmlc/mshadow/blob/master/mshadow/base.h
enum DataType {
  kF32 = 0,
  kF64 = 1,
  kF16 = 2,
  kU8 = 3,
  kI32 = 4,
  kI8 = 5,
  kI64 = 6,
};

#define CPU_DEVICE_ID (-1)
#define CPU_DEVICE_MMAP_ID (-2)

enum DeviceType { kCPU = 0, kMMAP = 1, kGPU = 2 };

struct Context {
  DeviceType device_type;
  int device_id;
};

using StreamHandle = void*;

class Tensor;
using TensorPtr = std::shared_ptr<Tensor>;

class Tensor {
 public:
  Tensor();
  ~Tensor();

  bool defined() const { return _data; }
  DataType dtype() const { return _dtype; }
  const std::vector<size_t>& shape() const { return _shape; }
  const void* data() const { return _data; }
  void* mutable_data() { return _data; }
  size_t size() const { return _size; }
  int device() const { return _device; }

  static TensorPtr Empty(DataType dtype, std::vector<size_t> shape, int device,
                         std::string name);
  static TensorPtr CreateCopy1D(TensorPtr tensor, size_t item_offset,
                                std::vector<size_t> shape, std::string name,
                                cudaStream_t stream = nullptr);
  static TensorPtr FromMmap(std::string filepath, DataType dtype,
                            std::vector<size_t> shape, int device,
                            std::string name, cudaStream_t stream = nullptr);
  static TensorPtr FromBlob(void* data, DataType dtype,
                            std::vector<size_t> shape, int device,
                            std::string name);

 private:
  void* _data;
  int _device;
  DataType _dtype;
  size_t _size;
  std::vector<size_t> _shape;
  std::string _name;
};

// Graph dataset that should be loaded from the .
struct Dataset {
  // Graph topology data
  TensorPtr indptr;
  TensorPtr indices;
  size_t num_node;
  size_t num_edge;

  // Node feature and label
  size_t num_class;
  TensorPtr feat;
  TensorPtr label;

  // Node set
  TensorPtr train_set;
  TensorPtr test_set;
  TensorPtr valid_set;
};

// Train graph format that is compatible with the cuSparse method
struct TrainGraph {
  TensorPtr row;
  TensorPtr col;
  TensorPtr val;
  size_t num_row;
  size_t num_column;
  size_t num_edge;
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
};

using GraphBatch = Task;

size_t GetDataTypeLength(int dtype);

std::string ToReadableSize(size_t size_in_bytes);

}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_COMMON_H
