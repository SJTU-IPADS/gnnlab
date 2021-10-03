#include <unordered_set>

#include "../common.h"
#include "../constant.h"
#include "../logging.h"
#include "cpu_function.h"

namespace samgraph {
namespace common {
namespace cpu {

void CPUSanityCheckList(const IdType *input, size_t num_input,
                        IdType invalid_val) {
  for (size_t i = 0; i < num_input; i++) {
    CHECK_NE(input[i], invalid_val);
  }
}

void CPUSanityCheckNoDuplicate(const IdType *input, size_t num_input) {
  std::unordered_set<IdType> visited_elem;
  for (size_t i = 0; i < num_input; i++) {
    if (visited_elem.count(input[i]) > 0) {
      LOG(DEBUG) << "duplicate" << input[i];
      CHECK(false);
    }
    visited_elem.insert(input[i]);
  }
}

}  // namespace cpu
}  // namespace common
}  // namespace samgraph