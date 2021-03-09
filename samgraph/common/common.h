#ifndef SAMGRAPH_COMMON_H
#define SAMGRAPH_COMMON_H

#include <string>
#include <memory>
#include <functional>
#include <vector>
#include <mutex>

namespace samgraph {
namespace common {

// Keep the order consistent with DMLC/mshadow
// https://github.com/dmlc/mshadow/blob/master/mshadow/base.h
enum DataType {
  kSamF32 = 0,
  kSamF64 = 1,
  kSamF16 = 2,
  kSamU8 = 3,
  kSamI32 = 4,
  kSamI8 = 5,
  kSamI64 = 6,
};

#define CPU_DEVICE_ID (-1)
#define CPU_DEVICE_MMAP_ID (-2)

class Tensor;

// Data Container is used to manage 
// the underlying data storage
class DataContainer {
 public:
  DataContainer(void *data, size_t size, int device)
    : _data(data), _size(size), _device(device) {}
  // Destroy the tensor when the _data is not null
  ~DataContainer();
  bool Consume();

 private:
  void *_data;
  size_t _size;
  int _device;

  friend class Tensor;
};

class Tensor {
 public:
  // Create a tensor from the file and take the
  // ownership of the input data
  static std::shared_ptr<Tensor> FromMmap(std::string filepath, DataType dtype,
                                          std::vector<size_t> dims, int device);
  // Create an uninitialized tensor data
  static std::shared_ptr<Tensor> Empty(DataType dtype, std::vector<size_t> dims, int device);
  // Create view from a tensor, they share the same storage
  static std::shared_ptr<Tensor> CreateView(std::shared_ptr<Tensor> tensor, size_t offset,
                                            std::vector<size_t> dims);
  // Deep slice a tensor
  static std::shared_ptr<Tensor> DeepCopy(std::shared_ptr<Tensor> tensor, size_t offset,
                                       std::vector<size_t> dims);
  // From blob
  static std::shared_ptr<Tensor> FromBlob(void* data, DataType dtype,
                                          std::vector<size_t> dims, int device);
  // Consume the tensor, and we don't have to free the data.
  bool ConsumeData() { return _container->Consume(); }

  bool defined() const { return _container && _container->_data; }
  DataType dtype() const { return _dtype; }
  const std::vector<size_t> &shape() const { return _dims; }
  const void* data() const { return static_cast<char*>(_container->_data) + _offset; }
  void* mutable_data() { return static_cast<char *>(_container->_data) + _offset; }
  size_t size() const { return _size; }
  int device() const { return _container->_device; }

 private:
  // Data container
  std::shared_ptr<DataContainer> _container;
  // Start offset
  size_t _offset;
  // Data type
  DataType _dtype;
  // Size of tensor in bytes
  size_t _size;
  // Tensor dimensions
  std::vector<size_t> _dims;
};

// IdTensors only have one dimesion
using IdTensor = Tensor;

// Graph dataset that should be loaded from the .
struct SamGraphDataset {
    // Graph topology data
    std::shared_ptr<IdTensor> indptr;
    std::shared_ptr<IdTensor> indices;
    size_t num_node;
    size_t num_edge;

    // Node feature and label
    size_t num_class;
    std::shared_ptr<Tensor> feat;
    std::shared_ptr<Tensor> label;

    // Node set
    std::shared_ptr<IdTensor> train_set;
    std::shared_ptr<IdTensor> test_set;
    std::shared_ptr<IdTensor> valid_set;
};

enum QueueType {
    ID_COPYH2D = 0,
    DEV_SAMPLE,
    GRAPH_COPYD2D,
    ID_COPYD2H,
    FEAT_EXTRACT,
    FEAT_COPYH2D,
    SUBMIT,
    QUEUE_NUM_AND_NOT_A_REAL_QUEUE_TYPE_AND_MUST_BE_THE_LAST
};

const int QueueNum =
    (int)QUEUE_NUM_AND_NOT_A_REAL_QUEUE_TYPE_AND_MUST_BE_THE_LAST;

// Train graph format that is compatible with the cuSparse method
struct TrainGraph {
  std::shared_ptr<IdTensor> indptr;
  std::shared_ptr<IdTensor> indices;
  int num_row;
  int num_column;
  int num_edge;
};

struct TaskEntry {
    // Key of the task
    uint64_t key;
    // Input tensor
    std::shared_ptr<IdTensor> train_nodes;
    // Current input tensor
    std::shared_ptr<IdTensor> cur_input;
    // Output graph tensor
    std::vector<std::shared_ptr<TrainGraph>> output_graph;
    // Original node ids of the first train graph
    std::shared_ptr<Tensor> output_nodes;
    // Output feature tensor
    std::shared_ptr<Tensor> output_feat;
    // Output label tensor
    std::shared_ptr<Tensor> output_label;
};

using GraphBatch = TaskEntry;

uint64_t encodeKey(int epoch_idx, size_t batch_idx);
int getDataTypeLength(int dtype);

} // namespace common
} // namespace samgraph

#endif // SAMGRAPH_COMMON_H
