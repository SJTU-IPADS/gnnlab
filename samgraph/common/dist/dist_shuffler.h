#ifndef SAMGRAPH_DIST_SHUFFLER_H
#define SAMGRAPH_DIST_SHUFFLER_H

#include <limits>
#include <memory>

#include "../common.h"

namespace samgraph {
namespace common {
namespace dist {

class DistShuffler : public Shuffler {
 public:
  DistShuffler(TensorPtr input, size_t num_epoch, size_t batch_size,
              int sampler_id, int num_sampler, int num_trainer,
              bool drop_last);
  TensorPtr GetBatch(StreamHandle stream = nullptr) override;

  uint64_t Epoch() override { return _cur_epoch; }
  uint64_t Step() override { return _cur_step; }

  size_t NumEpoch() override { return _num_epoch; }
  // return the total steps for each epoch
  // reasons: profiler needs this to create total space
  size_t NumStep() override { return _epoch_step; }

  void Reset() { _cur_step = _num_step; _cur_epoch = 0; _initialized = false; }
  // global key
  uint64_t GetBatchKey() {
    return _cur_epoch * _epoch_step +
           (_dataset_offset / _batch_size) + _cur_step; }

 private:
  bool _drop_last;
  bool _initialized;

  uint64_t _cur_epoch;
  uint64_t _cur_step;

  size_t _num_epoch;
  // number of steps for this sampler
  size_t _num_step;
  // total steps each epoch
  size_t _epoch_step;
  // the offset of train set for this sampler
  size_t _dataset_offset;

  TensorPtr _data;
  TensorPtr _gpu_data;
  size_t _num_data;

  size_t _batch_size;
  size_t _last_batch_size;

  IdType *_sanity_check_map;

  void ReShuffle(StreamHandle stream = nullptr);
};

}  // namespace dist
}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_DIST_SHUFFLER_H
