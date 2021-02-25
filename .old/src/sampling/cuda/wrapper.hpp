#pragma once

void sampleBlockWrapper(const unsigned int *indptr, const unsigned int *indices,
                        const unsigned int *input,
                        unsigned int *output,
                        unsigned int num_input,
                        unsigned int fanout,
                        unsigned int seed);