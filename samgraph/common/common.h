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

class Tensor;
using TensorPtr = std::shared_ptr<Tensor>;

class Tensor {
 public:
  Tensor();
  ~Tensor();
  // Create a tensor from the file and take the
  // ownership of the input data
  static TensorPtr FromMmap(std::string filepath, DataType dtype,
                            std::vector<size_t> shape, int device,
                            std::string name, cudaStream_t stream = nullptr);
  // Create an uninitialized tensor data
  static TensorPtr Empty(DataType dtype, std::vector<size_t> shape, int device,
                         std::string name);
  // Deep slice a tensor
  static TensorPtr CreateCopy1D(TensorPtr tensor, size_t item_offset,
                                std::vector<size_t> shape, std::string name,
                                cudaStream_t stream = nullptr);
  // From blob
  static TensorPtr FromBlob(void* data, DataType dtype,
                            std::vector<size_t> shape, int device,
                            std::string name);
  static TensorPtr ToDevice(const TensorPtr origin, int device,
                            cudaStream_t stream = nullptr);

  bool defined() const { return _data; }
  DataType dtype() const { return _dtype; }
  const std::vector<size_t>& shape() const { return _shape; }
  const void* data() const { return _data; }
  void* mutable_data() { return _data; }
  size_t size() const { return _size; }
  int device() const { return _device; }

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
  int num_row;
  int num_column;
  int num_edge;
};

struct Task {
  // Key of the task
  uint64_t key;
  // Current input tensor
  TensorPtr cur_input;
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

uint64_t EncodeBatchKey(uint64_t epoch_idx, uint64_t batch_idx);
uint64_t DecodeBatchKey(uint64_t key);
uint64_t EncodeGraphId(uint64_t key, uint64_t graph_id);
uint64_t DecodeGraphId(uint64_t key);

size_t GetDataTypeLength(int dtype);

std::string ToReadableSize(size_t size_in_bytes);

}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_COMMON_H
