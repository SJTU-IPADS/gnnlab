#ifndef SAMGRAPH_COMMON_H
#define SAMGRAPH_COMMON_H

#include <string>
#include <memory>
#include <functional>
#include <vector>
#include <mutex>
#include <cstdint>

#include <cuda_runtime.h>

namespace samgraph {
namespace common {

using IdType = unsigned int;
using SignedIdType = int;

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

class Tensor {
 public:
  Tensor();
  ~Tensor();
  // Create a tensor from the file and take the
  // ownership of the input data
  static std::shared_ptr<Tensor> FromMmap(std::string filepath, DataType dtype,
                                          std::vector<size_t> shape, int device,
                                          std::string name, cudaStream_t stream = nullptr);
  // Create an uninitialized tensor data
  static std::shared_ptr<Tensor> Empty(DataType dtype, std::vector<size_t> shape, int device, std::string name);
  // Deep slice a tensor
  static std::shared_ptr<Tensor> CreateCopy1D(std::shared_ptr<Tensor> tensor, size_t item_offset,
                                       std::vector<size_t> shape, std::string name, cudaStream_t stream = nullptr);
  // From blob
  static std::shared_ptr<Tensor> FromBlob(void* data, DataType dtype,
                                          std::vector<size_t> shape, int device, std::string name);
  static std::shared_ptr<Tensor> ToDevice(const std::shared_ptr<Tensor> origin, int device, cudaStream_t stream = nullptr);

  bool defined() const { return _data; }
  DataType dtype() const { return _dtype; }
  const std::vector<size_t> &shape() const { return _shape; }
  const void* data() const { return _data; }
  void* mutable_data() { return _data; }
  size_t size() const { return _size; }
  int device() const { return _device; }

 private:
  void *_data;
  int _device;
  DataType _dtype;
  size_t _size;
  std::vector<size_t> _shape;
  std::string _name;
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

enum CudaQueueType {
    CUDA_SAMPLE,
    CUDA_GRAPH_COPYD2D,
    CUDA_ID_COPYD2H,
    CUDA_FEAT_EXTRACT,
    CUDA_FEAT_COPYH2D,
    CUDA_SUBMIT,
    CUDA_QUEUE_NUM_AND_NOT_A_REAL_QUEUE_TYPE_AND_MUST_BE_THE_LAST
};

const int CudaQueueNum =
    (int)CUDA_QUEUE_NUM_AND_NOT_A_REAL_QUEUE_TYPE_AND_MUST_BE_THE_LAST;

// Train graph format that is compatible with the cuSparse method
struct TrainGraph {
  std::shared_ptr<IdTensor> indptr;
  std::shared_ptr<IdTensor> indices;
  std::shared_ptr<IdTensor> val;
  int num_row;
  int num_column;
  int num_edge;
};

struct TaskEntry {
    uint64_t epoch;
    uint64_t step;
    // Key of the task
    uint64_t key;
    // Train nodes
    std::shared_ptr<IdTensor> train_nodes;
    // Current input tensor
    std::shared_ptr<IdTensor> cur_input;
    // Output graph tensor
    std::vector<std::shared_ptr<TrainGraph>> output_graph;
    // node ids of the last train graph
    std::shared_ptr<IdTensor> input_nodes;
    // node ids of the first train graph
    std::shared_ptr<IdTensor> output_nodes;
    // Input feature tensor
    std::shared_ptr<Tensor> input_feat;
    // Output label tensor
    std::shared_ptr<Tensor> output_label;
};

using GraphBatch = TaskEntry;

uint64_t encodeBatchKey(uint64_t epoch_idx, uint64_t batch_idx);
uint64_t decodeBatchKey(uint64_t key);
uint64_t encodeGraphID(uint64_t key, uint64_t graph_id);
uint64_t decodeGraphID(uint64_t key);

size_t getDataTypeLength(int dtype);

std::string toReadableSize(size_t size_in_bytes);

} // namespace common
} // namespace samgraph

#endif // SAMGRAPH_COMMON_H
