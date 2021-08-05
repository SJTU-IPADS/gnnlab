#ifndef SAMGRAPH_CUDA_FUNCTION_H
#define SAMGRAPH_CUDA_FUNCTION_H

#include "../common.h"
#include "cuda_frequency_hashmap.h"
#include "cuda_hashtable.h"
#include "cuda_random_states.h"

namespace samgraph {
namespace common {
namespace cuda {

void GPUSampleKHop0(const IdType *indptr, const IdType *indices,
                    const IdType *input, const size_t num_input,
                    const size_t fanout, IdType *out_src, IdType *out_dst,
                    size_t *num_out, Context ctx, StreamHandle stream,
                    GPURandomStates *random_states, uint64_t task_key);

void GPUSampleKHop1(const IdType *indptr, const IdType *indices,
                    const IdType *input, const size_t num_input,
                    const size_t fanout, IdType *out_src, IdType *out_dst,
                    size_t *num_out, Context ctx, StreamHandle stream,
                    GPURandomStates *random_states, uint64_t task_key);

void GPUSampleWeightedKHop(const IdType *indptr, const IdType *indices,
                           const float *prob_table, const IdType *alias_table,
                           const IdType *input, const size_t num_input,
                           const size_t fanout, IdType *out_src,
                           IdType *out_dst, size_t *num_out, Context ctx,
                           StreamHandle stream, GPURandomStates *random_states,
                           uint64_t task_key);

void GPUSampleRandomWalk(const IdType *indptr, const IdType *indices,
                         const IdType *input, const size_t num_input,
                         const size_t random_walk_length,
                         const double random_walk_restart_prob,
                         const size_t num_random_walk, const size_t K,
                         IdType *out_src, IdType *out_dst, IdType *out_data,
                         size_t *num_out, FrequencyHashmap *frequency_hashmap,
                         Context ctx, StreamHandle stream,
                         GPURandomStates *random_states, uint64_t task_key);

void GPUMapEdges(const IdType *const global_src, IdType *const new_global_src,
                 const IdType *const global_dst, IdType *const new_global_dst,
                 const size_t num_edges, DeviceOrderedHashTable mapping,
                 Context ctx, StreamHandle stream);

void GPUExtract(void *dst, const void *src, const IdType *index,
                size_t num_index, size_t dim, DataType dtype, Context ctx,
                StreamHandle stream, uint64_t task_key);

void GPUExtractNeighbour(const IdType *indptr, const IdType *indices,
                    const IdType *input, const size_t num_input,
                    IdType *&output,
                    size_t *num_out, Context ctx, StreamHandle stream,
                    const uint64_t task_key);

void GPUBatchSanityCheck(IdType *map, const IdType *input,
                         const size_t num_input, Context ctx,
                         StreamHandle stream);

void GPUSanityCheckList(const IdType *input, size_t num_input,
                        IdType invalid_val, Context ctx, StreamHandle stream);

}  // namespace cuda
}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_CUDA_FUNCTION_H
