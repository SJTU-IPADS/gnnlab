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

#ifndef SAMGRAPH_CPU_FUNCTION_H
#define SAMGRAPH_CPU_FUNCTION_H

#include "../common.h"

namespace samgraph {
namespace common {
namespace cpu {

void CPUSampleKHop0(const IdType *const indptr, const IdType *const indices,
                    const IdType *const input, const size_t num_input,
                    IdType *output_src, IdType *output_dst, size_t *num_ouput,
                    const size_t fanout);

void CPUSampleKHop1(const IdType *const indptr, const IdType *const indices,
                    const IdType *const input, const size_t num_input,
                    IdType *output_src, IdType *output_dst, size_t *num_ouput,
                    const size_t fanout);

void CPUSampleKHop2(const IdType *const indptr, IdType *indices,
                    const IdType *const input, const size_t num_input,
                    IdType *output_src, IdType *output_dst, size_t *num_ouput,
                    const size_t fanout);

void CPUSampleWeightedKHop(const IdType *const indptr,
                           const IdType *const indices,
                           const IdType *const input, const size_t num_input,
                           IdType *output_src, IdType *output_dst,
                           size_t *num_ouput, const size_t fanout);

void CPUSampleRandomWalk(const IdType *const indptr,
                         const IdType *const indices, const IdType *const input,
                         const size_t num_input, IdType *output_src,
                         IdType *output_dst, size_t *num_ouput,
                         const size_t fanout);

void CPUExtract(void *dst, const void *src, const IdType *index,
                size_t num_index, size_t dim, DataType dtype);

void CPUMockExtract(void *dst, const void *src, const IdType *index,
                size_t num_index, size_t dim, DataType dtype);

IdType RandomID(const IdType &min, const IdType &max);

void CPUSanityCheckList(const IdType *input, size_t num_input,
                        IdType invalid_val);

void CPUSanityCheckNoDuplicate(const IdType *input, size_t num_input);

}  // namespace cpu
}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_CPU_FUNCTION_H