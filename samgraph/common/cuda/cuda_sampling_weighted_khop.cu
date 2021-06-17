#include "cuda_function.h"

namespace samgraph {
namespace common {
namespace cuda {

void GPUSampleWeighted(const IdType *indptr, const IdType *indices,
                       const IdType *input, const size_t num_input,
                       const size_t fanout, IdType *out_src, IdType *out_dst,
                       size_t *num_out, Context ctx, StreamHandle stream,
                       GPURandomStates *random_states, uint64_t task_key) {}

}  // namespace cuda
}  // namespace common
}  // namespace samgraph