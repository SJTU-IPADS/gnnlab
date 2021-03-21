#include "cpu_function.h"

namespace samgraph {
namespace common {
namespace cpu {

void ConvertCoo2Csr(const IdType *src, const IdType *dst,
                    IdType *out_indptr, IdType *out_indices,
                    const size_t num_row, const size_t num_edge) {
    // compute indptr
    IdType *Bp = out_indptr;
    *(Bp++) = 0;
    std::fill(Bp, Bp + num_row, 0);
    for (size_t i = 0; i < num_edge; ++i) {
      Bp[src[i]]++;
    }

    // prefix sum
    IdType prefix_sum = 0;
    for (size_t i = 0; i < num_row; ++i) {
      const IdType temp = Bp[i];
      Bp[i] = prefix_sum;
      prefix_sum += temp;
    }

    IdType *Bi = out_indices;

    // compute indices and data
    for (size_t i = 0; i < num_edge; ++i) {
      const IdType r = src[i];
      Bi[Bp[r]++] = dst[i];
    }
}

} // namespace cpu
} // namespace common
} // namespace samgraph
