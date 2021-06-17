#include <random>

#include "cpu_function.h"

namespace samgraph {
namespace common {
namespace cpu {

IdType RandomID(const IdType &min, const IdType &max) {
  static thread_local std::mt19937 generator;
  std::uniform_int_distribution<IdType> distribution(min, max);
  return distribution(generator);
}

}  // namespace cpu
}  // namespace common
}  // namespace samgraph