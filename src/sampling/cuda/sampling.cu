#include <curand.h>

__global__ void sampleBlock(const unsigned int *indptr,
                            const unsigned int *indices,
                            const unsigned int *seeds,
                            const unsigned int *output,
                            unsigned int num_seeds,
                            int fanout) {
    int x;
    int y;

    if (x < num_seeds && y < fanout) {

    }
}
