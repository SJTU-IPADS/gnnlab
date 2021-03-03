#ifndef SAMGRAPH_RANDOM_PERMUTATION_H
#define SAMGRAPH_RANDOM_PERMUTATION_H

#include "types.h"
#include "common.h"

namespace samgraph {
namespace common {

class RandomPermutation {
 public:
  RandomPermutation(std::shared_ptr<IdTensor> input, int batch_size)
    : _tensor(input), _batch_size(batch_size) {}

  void Permutate();

 private:
  std::shared_ptr<IdTensor> _tensor;
  int _batch_size;
  int _max_num_batch;
  int _next_batch_idx;
};

} // namespace common
} // namespace samgraph

#endif // SAMGRAPH_RANDOM_PERMUTATION_H
