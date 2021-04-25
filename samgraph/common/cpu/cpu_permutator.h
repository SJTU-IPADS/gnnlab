#ifndef SAMGRAPH_CPU_PERMUTATOR_H
#define SAMGRAPH_CPU_PERMUTATOR_H

#include <limits>
#include <memory>

#include "../common.h"

namespace samgraph {
namespace common {

class CPUPermutator {
 public:
  CPUPermutator(TensorPtr input, int num_epoch, size_t batch_size,
                bool drop_last);
  TensorPtr GetBatch();

  uint64_t Epoch() { return _cur_epoch; }
  uint64_t Step() { return _cur_step; }

  size_t NumEpoch() { return _num_epoch; }
  size_t NumStep() { return _num_step; }

 private:
  bool _drop_last;
  bool _initialized;

  uint64_t _cur_epoch;
  uint64_t _cur_step;

  size_t _num_epoch;
  size_t _num_step;

  TensorPtr _data;
  size_t _num_data;

  size_t _batch_size;
  size_t _last_batch_size;

  void RePermutate();
};

}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_CPU_PERMUTATOR_H
