#ifndef SAMGRAPH_RANDOM_PERMUTATION_H
#define SAMGRAPH_RANDOM_PERMUTATION_H

#include <memory>

#include "types.h"
#include "common.h"

namespace samgraph {
namespace common {

class RandomPermutation {
 public:
  RandomPermutation(std::shared_ptr<IdTensor> input, int num_epoch,
                    size_t batch_size, bool drop_last);
  std::shared_ptr<IdTensor> GetBatch();

  inline int cur_epoch() { return _cur_epoch; }
  inline int num_epoch() { return _num_epoch; }
  inline size_t cur_batch() { return _cur_batch_idx; }
  inline size_t num_batch() { return _num_batch; }
  inline size_t num_element() { return _input->shape().front(); }

 private:
  int _num_epoch;
  int _cur_epoch;

  std::shared_ptr<IdTensor> _input;
  size_t _num_element;
  bool _drop_last;
  size_t _batch_size;
  size_t _last_batch_size;
  size_t _num_batch;
  size_t _cur_batch_idx;

  void Permutate();
};

} // namespace common
} // namespace samgraph

#endif // SAMGRAPH_RANDOM_PERMUTATION_H
