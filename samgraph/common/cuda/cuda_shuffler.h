#ifndef SAMGRAPH_CUDA_SHUFFLER_H
#define SAMGRAPH_CUDA_SHUFFLER_H

#include <limits>
#include <memory>

#include "../common.h"
#include "../partition.h"

namespace samgraph {
namespace common {
namespace cuda {

class GPUShuffler : public Shuffler {
 public:
  GPUShuffler(TensorPtr input, size_t num_epoch, size_t batch_size,
              bool drop_last);
  TensorPtr GetBatch(StreamHandle stream = nullptr);

  uint64_t Epoch() { return _cur_epoch; }
  uint64_t Step() { return _cur_step; }

  size_t NumEpoch() { return _num_epoch; }
  size_t NumStep() { return _num_step; }

  void Reset() { _cur_step = _num_step; _cur_epoch = 0; _initialized = false; }

 private:
  bool _drop_last;
  bool _initialized;

  uint64_t _cur_epoch;
  uint64_t _cur_step;

  size_t _num_epoch;
  size_t _num_step;

  TensorPtr _data;
  TensorPtr _gpu_data;
  size_t _num_data;

  size_t _batch_size;
  size_t _last_batch_size;

  IdType *_sanity_check_map;

  void ReShuffle(StreamHandle stream = nullptr);
};

class PartitionShuffler : public Shuffler {
 public:
  PartitionShuffler(PaGraphPartition&partition, 
    size_t num_epoch, size_t batch_size, bool drop_last);

  TensorPtr GetBatch(StreamHandle stream = nullptr) override;
  uint64_t Epoch() override { return _cur_epoch; }
  uint64_t Step() override { return _cur_step; } 

  size_t NumEpoch() override { return _num_epoch; }
  size_t NumStep() override { return _num_step; }

  void Reset() override;
 private:
  bool _drop_last;
  uint64_t _cur_epoch;
  uint64_t _cur_step;

  size_t _num_epoch;
  size_t _num_step;

  std::vector<std::unique_ptr<Shuffler>> _shufflers;
};

}  // namespace cuda
}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_CUDA_SHUFFLER_H
