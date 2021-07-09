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

}  // namespace cpu
}  // namespace common
}  // namespace samgraph