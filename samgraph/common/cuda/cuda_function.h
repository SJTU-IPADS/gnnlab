/*
 * Copyright 2022 Institute of Parallel and Distributed Systems, Shanghai Jiao Tong University
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

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

void GPUSampleWeightedKHopPrefix(const IdType *indptr, const IdType *indices,
                           const float *prob_prefix_table,
                           const IdType *input, const size_t num_input,
                           const size_t fanout, IdType *out_src,
                           IdType *out_dst, size_t *num_out, Context ctx,
                           StreamHandle stream, GPURandomStates *random_states,
                           uint64_t task_key);
void GPUSampleKHop2(const IdType *indptr, IdType *indices,
                    const IdType *input, const size_t num_input,
                    const size_t fanout, IdType *out_src, IdType *out_dst,
                    size_t *num_out, Context ctx, StreamHandle stream,
                    GPURandomStates *random_states, uint64_t task_key);
void GPUSampleWeightedKHopHashDedup(const IdType *indptr, const IdType *indices,
                           const float *prob_table, const IdType *alias_table,
                           const IdType *input, const size_t num_input,
                           const size_t fanout, IdType *out_src,
                           IdType *out_dst, size_t *num_out, Context ctx,
                           StreamHandle stream, GPURandomStates *random_states,
                           uint64_t task_key);

void GPUMapEdges(const IdType *const global_src, IdType *const new_global_src,
                 const IdType *const global_dst, IdType *const new_global_dst,
                 const size_t num_edges, DeviceOrderedHashTable mapping,
                 Context ctx, StreamHandle stream);

void GPUExtract(void *dst, const void *src, const IdType *index,
                size_t num_index, size_t dim, DataType dtype, Context ctx,
                StreamHandle stream, uint64_t task_key);

void GPUMockExtract(void *dst, const void *src, const IdType *index,
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

void GetMissCacheIndex(
    IdType *sampler_gpu_hashtable, Context sampler_ctx,
    IdType *output_miss_src_index, IdType *output_miss_dst_index,
    size_t *num_output_miss, IdType *output_cache_src_index,
    IdType *output_cache_dst_index, size_t *num_output_cache,
    const IdType *nodes, const size_t num_nodes, StreamHandle stream);

}  // namespace cuda
}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_CUDA_FUNCTION_H
